"""LLM provider protocol.

Defines the minimal interface any LLM backend must satisfy.
Keeps the framework decoupled from specific SDKs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """Structured response from an LLM call."""

    content: str
    model: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    raw: Any = None


class LLMProvider(Protocol):
    """Protocol for LLM backends."""

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str = "",
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> LLMResponse: ...
