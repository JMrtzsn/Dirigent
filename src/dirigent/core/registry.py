"""Agent registry — capability-based discovery and routing.

The registry maps capabilities to agents. When a DelegateRequest arrives
targeting a capability, the registry resolves which agent handles it.
"""

from __future__ import annotations

from dirigent.core.agent import Agent, AgentInfo


class RegistryError(Exception):
    """Raised when capability lookup fails."""


class Registry:
    """In-memory agent registry with capability-based lookup."""

    def __init__(self) -> None:
        self._agents: dict[str, Agent] = {}

    def register(self, agent: Agent) -> None:
        """Register an agent by its name."""
        info = agent.info
        if info.name in self._agents:
            msg = f"Agent '{info.name}' already registered"
            raise RegistryError(msg)
        self._agents[info.name] = agent

    def resolve_capability(self, capability: str) -> Agent:
        """Find an agent that provides the requested capability."""
        for ag in self._agents.values():
            for cap in ag.info.capabilities:
                if cap.matches(capability):
                    return ag
        msg = f"No agent provides capability '{capability}'"
        raise RegistryError(msg)

    def resolve_name(self, name: str) -> Agent:
        """Find an agent by name."""
        if name not in self._agents:
            msg = f"Agent '{name}' not found"
            raise RegistryError(msg)
        return self._agents[name]

    def list_agents(self) -> list[AgentInfo]:
        """Return info for all registered agents."""
        return [ag.info for ag in self._agents.values()]

    def list_capabilities(self) -> list[str]:
        """Return all registered capability names."""
        caps: list[str] = []
        for ag in self._agents.values():
            caps.extend(c.name for c in ag.info.capabilities)
        return caps
