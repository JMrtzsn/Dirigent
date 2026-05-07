"""Fan-out pattern — send the same task to N agents, collect all results."""

from __future__ import annotations

import asyncio
from typing import Any

from dirigent.core.message import DelegateResponse
from dirigent.runtime import Runtime


async def fanout(
    runtime: Runtime,
    *,
    capability: str = "",
    agents: list[str] | None = None,
    task: str,
    context: dict[str, Any] | None = None,
    constraints: dict[str, Any] | None = None,
    sender: str = "__fanout__",
) -> list[DelegateResponse]:
    """Fan out a task to multiple agents and collect all responses.

    Either provide `agents` (list of agent names) or `capability` to fan out
    to all agents that provide that capability.
    """
    if agents:
        targets = agents
    elif capability:
        # Find all agents with this capability
        targets = [
            ag.name
            for ag in runtime.registry.list_agents()
            if any(c.matches(capability) for c in ag.capabilities)
        ]
        if not targets:
            from dirigent.core.registry import RegistryError

            raise RegistryError(f"No agents provide capability '{capability}'")
    else:
        msg = "Must provide either 'agents' or 'capability'"
        raise ValueError(msg)

    coros = [
        runtime.delegate(
            to=name,
            task=task,
            context=context,
            constraints=constraints,
            sender=sender,
        )
        for name in targets
    ]

    return list(await asyncio.gather(*coros))
