"""Runtime — the execution engine that connects agents via the registry.

The runtime is the single entry point for dispatching work. It resolves
capabilities, delivers messages, enforces timeouts, and collects traces.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from dirigent.core.agent import Agent
from dirigent.core.message import DelegateRequest, DelegateResponse
from dirigent.core.registry import Registry


@dataclass
class RuntimeConfig:
    """Configuration for the runtime."""

    timeout_seconds: float = 300.0
    max_delegation_depth: int = 10


class DelegationDepthError(Exception):
    """Raised when agents delegate beyond the configured depth limit."""


class Runtime:
    """Orchestrates agent execution via capability-based routing."""

    def __init__(self, config: RuntimeConfig | None = None) -> None:
        self._registry = Registry()
        self._config = config or RuntimeConfig()
        self._traces: list[dict[str, Any]] = []
        self._depth: int = 0

    @property
    def registry(self) -> Registry:
        return self._registry

    @property
    def traces(self) -> list[dict[str, Any]]:
        """Return recorded message traces."""
        return list(self._traces)

    def register(self, *agents: Agent) -> None:
        """Register one or more agents."""
        for ag in agents:
            self._registry.register(ag)

    async def delegate(
        self,
        *,
        capability: str = "",
        to: str = "",
        task: str,
        context: dict[str, Any] | None = None,
        constraints: dict[str, Any] | None = None,
        sender: str = "__runtime__",
    ) -> DelegateResponse:
        """Delegate work to an agent by capability or name.

        Either `capability` or `to` must be provided. If both are given,
        `to` (direct name) takes precedence.
        """
        if not capability and not to:
            msg = "Must provide either 'capability' or 'to'"
            raise ValueError(msg)

        if self._depth >= self._config.max_delegation_depth:
            raise DelegationDepthError(
                f"Delegation depth {self._depth} exceeds limit {self._config.max_delegation_depth}"
            )

        # Resolve target agent
        if to:
            agent = self._registry.resolve_name(to)
        else:
            agent = self._registry.resolve_capability(capability)

        request = DelegateRequest(
            sender=sender,
            receiver=agent.info.name,
            capability=capability or to,
            task=task,
            context=context or {},
            constraints=constraints or {},
        )

        # Record trace
        trace_entry = {
            "request_id": request.id,
            "sender": request.sender,
            "receiver": request.receiver,
            "capability": request.capability,
            "task": request.task,
        }

        self._depth += 1
        try:
            response = await asyncio.wait_for(
                agent.handle(request),
                timeout=self._config.timeout_seconds,
            )
        except TimeoutError:
            response = DelegateResponse(
                sender=agent.info.name,
                receiver=sender,
                correlation_id=request.id,
                success=False,
                error=f"Timed out after {self._config.timeout_seconds}s",
            )
        finally:
            self._depth -= 1

        trace_entry["response_id"] = response.id
        trace_entry["success"] = response.success
        trace_entry["error"] = response.error
        self._traces.append(trace_entry)

        return response
