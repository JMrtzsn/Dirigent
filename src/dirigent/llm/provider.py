"""LLM provider protocol — the contract all providers must satisfy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class Message:
    """A single message in a conversation."""

    role: str  # "system", "user", "assistant"
    content: str


@dataclass
class CompletionResult:
    """Response from an LLM completion call."""

    content: str
    model: str
    usage: dict[str, int] = field(default_factory=dict)


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol that all LLM providers must implement.

    Kept minimal on purpose — one method, no streaming (yet).
    Streaming can be added as a separate method when needed.
    """

    def complete(
        self,
        messages: list[Message],
        *,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> CompletionResult:
        """Send messages to an LLM and return the response.

        Args:
            messages: Conversation history.
            model: Model identifier (e.g. "claude-sonnet-4.6").
            temperature: Sampling temperature. 0.0 = deterministic.
            max_tokens: Maximum tokens in the response.

        Returns:
            CompletionResult with the model's response.

        Raises:
            ProviderError: If the API call fails.
        """
        ...


class ProviderError(Exception):
    """Raised when an LLM provider call fails."""

    def __init__(self, message: str, *, provider: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code
