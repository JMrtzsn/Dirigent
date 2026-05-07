"""Metrics — cost, latency, and token tracking per agent."""

from __future__ import annotations

import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any


@dataclass
class AgentMetrics:
    """Accumulated metrics for a single agent."""

    calls: int = 0
    successes: int = 0
    failures: int = 0
    total_latency_ms: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.calls if self.calls else 0.0

    @property
    def success_rate(self) -> float:
        return self.successes / self.calls if self.calls else 0.0


class Metrics:
    """Collects per-agent metrics across a runtime session."""

    def __init__(self) -> None:
        self._agents: dict[str, AgentMetrics] = {}

    def _ensure(self, agent_name: str) -> AgentMetrics:
        if agent_name not in self._agents:
            self._agents[agent_name] = AgentMetrics()
        return self._agents[agent_name]

    def record_call(
        self,
        agent_name: str,
        *,
        success: bool,
        latency_ms: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        m = self._ensure(agent_name)
        m.calls += 1
        if success:
            m.successes += 1
        else:
            m.failures += 1
        m.total_latency_ms += latency_ms
        m.total_input_tokens += input_tokens
        m.total_output_tokens += output_tokens

    @contextmanager
    def measure(self, agent_name: str) -> Generator[dict[str, Any], None, None]:
        """Context manager that auto-records latency. Caller sets result['success']."""
        result: dict[str, Any] = {"success": True, "input_tokens": 0, "output_tokens": 0}
        start = time.perf_counter()
        try:
            yield result
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self.record_call(
                agent_name,
                success=result["success"],
                latency_ms=elapsed_ms,
                input_tokens=result.get("input_tokens", 0),
                output_tokens=result.get("output_tokens", 0),
            )

    def get(self, agent_name: str) -> AgentMetrics:
        return self._ensure(agent_name)

    def summary(self) -> dict[str, Any]:
        """Return a summary dict of all agent metrics."""
        return {
            name: {
                "calls": m.calls,
                "success_rate": round(m.success_rate, 3),
                "avg_latency_ms": round(m.avg_latency_ms, 1),
                "total_tokens": m.total_input_tokens + m.total_output_tokens,
            }
            for name, m in self._agents.items()
        }
