"""Capability declaration and matching.

A capability is a named unit of work an agent can perform.
Routing is capability-based: callers request a capability, the registry
finds an agent that provides it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Capability:
    """Declares something an agent can do."""

    name: str
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] = field(default_factory=dict)

    def matches(self, requested: str) -> bool:
        """Check if this capability satisfies a request (exact match for now)."""
        return self.name == requested
