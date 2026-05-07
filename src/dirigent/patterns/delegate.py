"""Delegate pattern — simple request/response delegation via the runtime."""

from __future__ import annotations

from typing import Any

from dirigent.core.message import DelegateResponse
from dirigent.runtime import Runtime


async def delegate(
    runtime: Runtime,
    *,
    capability: str = "",
    to: str = "",
    task: str,
    context: dict[str, Any] | None = None,
    constraints: dict[str, Any] | None = None,
    sender: str = "__pattern__",
) -> DelegateResponse:
    """Delegate a task to a single agent. Thin wrapper over runtime.delegate."""
    return await runtime.delegate(
        capability=capability,
        to=to,
        task=task,
        context=context,
        constraints=constraints,
        sender=sender,
    )
