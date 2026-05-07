"""Agent protocol — identity, capabilities, message handling.

An Agent is anything that can receive a DelegateRequest and return a DelegateResponse.
The @agent decorator provides a convenient way to declare capabilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from dirigent.core.capability import Capability
from dirigent.core.message import DelegateRequest, DelegateResponse


@dataclass(frozen=True, slots=True)
class AgentInfo:
    """Static metadata about an agent."""

    name: str
    capabilities: list[Capability] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Agent(Protocol):
    """Protocol that all agents must satisfy."""

    @property
    def info(self) -> AgentInfo: ...

    async def handle(self, request: DelegateRequest) -> DelegateResponse: ...


def agent(
    *,
    name: str | None = None,
    capabilities: list[str] | None = None,
    description: str = "",
):
    """Class decorator that wires up AgentInfo from declarative metadata.

    Usage:
        @agent(capabilities=["plan_architecture", "decompose_feature"])
        class Architect:
            async def handle(self, request: DelegateRequest) -> DelegateResponse:
                ...
    """

    def decorator(cls: type) -> type:
        agent_name = name or cls.__name__.lower()
        caps = [Capability(name=c) for c in (capabilities or [])]
        _info = AgentInfo(name=agent_name, capabilities=caps)

        # Inject info property
        cls.info = property(lambda self: _info)  # type: ignore[attr-defined]

        # Validate handle method exists
        if not hasattr(cls, "handle"):
            msg = f"@agent class {cls.__name__} must define an async handle() method"
            raise TypeError(msg)

        return cls

    return decorator
