"""GitHub Copilot provider implementing the new framework's LLMProvider protocol.

Reuses the token resolution logic from the legacy provider. Async interface
backed by OpenAI's sync client run in a thread executor.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from functools import partial
from pathlib import Path

from openai import OpenAI

from dirigent.llm.provider import LLMResponse

logger = logging.getLogger(__name__)

COPILOT_BASE_URL = "https://api.githubcopilot.com"
OPENCODE_AUTH_PATH = Path.home() / ".local" / "share" / "opencode" / "auth.json"

_COPILOT_HEADERS = {
    "Openai-Intent": "conversation-edits",
    "x-initiator": "user",
}


class CopilotProviderError(Exception):
    """Raised when Copilot API calls fail."""


def _resolve_token() -> str:
    """Resolve a Copilot OAuth token from env var or OpenCode auth cache."""
    env_token = os.environ.get("DIRIGENT_COPILOT_TOKEN")
    if env_token:
        return env_token

    if OPENCODE_AUTH_PATH.exists():
        try:
            data = json.loads(OPENCODE_AUTH_PATH.read_text())
            token = data.get("github-copilot", {}).get("refresh", "")
            if token:
                return token
        except (json.JSONDecodeError, KeyError, OSError) as exc:
            logger.warning("Failed to read OpenCode auth cache: %s", exc)

    raise CopilotProviderError(
        "No Copilot token found. Set DIRIGENT_COPILOT_TOKEN or authenticate OpenCode first."
    )


class CopilotLLMProvider:
    """Async LLM provider for GitHub Copilot (OpenAI-compatible API).

    Satisfies the dirigent.llm.provider.LLMProvider protocol.

    Usage:
        provider = CopilotLLMProvider()
        agent = LLMAgent(name="planner", provider=provider, model="claude-sonnet-4.6")
    """

    def __init__(self, *, token: str | None = None) -> None:
        resolved_token = token or _resolve_token()
        self._client = OpenAI(
            api_key=resolved_token,
            base_url=COPILOT_BASE_URL,
            default_headers={
                **_COPILOT_HEADERS,
                "User-Agent": "Dirigent/0.2.0",
            },
        )

    def _sync_complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        """Synchronous completion call (runs in thread pool)."""
        try:
            response = self._client.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore[arg-type]
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            raise CopilotProviderError(f"Copilot API call failed: {exc}") from exc

        choice = response.choices[0]
        content = choice.message.content or ""

        usage: dict[str, int] = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return LLMResponse(
            content=content,
            model=response.model,
            usage=usage,
            raw=response,
        )

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str = "claude-sonnet-4.6",
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Async completion — delegates to sync client in a thread executor."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            partial(
                self._sync_complete,
                messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            ),
        )
