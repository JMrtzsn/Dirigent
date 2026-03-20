"""GitHub Copilot LLM provider.

Authenticates via OpenCode's cached OAuth token or a DIRIGENT_COPILOT_TOKEN env var.
Uses the OpenAI-compatible API at api.githubcopilot.com.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from openai import OpenAI

from dirigent.llm.provider import CompletionResult, Message, ProviderError

logger = logging.getLogger(__name__)

COPILOT_BASE_URL = "https://api.githubcopilot.com"
OPENCODE_AUTH_PATH = Path.home() / ".local" / "share" / "opencode" / "auth.json"

# Required headers for the Copilot API
_COPILOT_HEADERS = {
    "Openai-Intent": "conversation-edits",
    "x-initiator": "user",
}


def _resolve_token() -> str:
    """Resolve a Copilot OAuth token.

    Resolution order:
    1. DIRIGENT_COPILOT_TOKEN env var
    2. OpenCode's cached auth at ~/.local/share/opencode/auth.json

    Returns:
        The OAuth token string.

    Raises:
        ProviderError: If no token can be found.
    """
    # 1. Env var override
    env_token = os.environ.get("DIRIGENT_COPILOT_TOKEN")
    if env_token:
        logger.debug("Using Copilot token from DIRIGENT_COPILOT_TOKEN env var")
        return env_token

    # 2. OpenCode's cached auth
    if OPENCODE_AUTH_PATH.exists():
        try:
            data = json.loads(OPENCODE_AUTH_PATH.read_text())
            copilot_auth = data.get("github-copilot", {})
            # OpenCode stores the usable token in the "refresh" field
            token = copilot_auth.get("refresh", "")
            if token:
                logger.debug("Using Copilot token from OpenCode auth cache")
                return token
        except (json.JSONDecodeError, KeyError, OSError) as exc:
            logger.warning("Failed to read OpenCode auth cache: %s", exc)

    raise ProviderError(
        "No Copilot token found. Either:\n"
        "  1. Set DIRIGENT_COPILOT_TOKEN env var\n"
        "  2. Authenticate OpenCode first (opencode → /connect → github-copilot)\n"
        f"  Looked for auth at: {OPENCODE_AUTH_PATH}",
        provider="copilot",
    )


class CopilotProvider:
    """GitHub Copilot LLM provider using the OpenAI-compatible API.

    Usage:
        provider = CopilotProvider()
        result = provider.complete(
            messages=[Message(role="user", content="Hello")],
            model="claude-sonnet-4.6",
        )
        print(result.content)
    """

    def __init__(self, *, token: str | None = None) -> None:
        """Initialize the Copilot provider.

        Args:
            token: Explicit OAuth token. If not provided, resolves automatically.
        """
        resolved_token = token or _resolve_token()
        self._client = OpenAI(
            api_key=resolved_token,
            base_url=COPILOT_BASE_URL,
            default_headers={
                **_COPILOT_HEADERS,
                "User-Agent": "Dirigent/0.1.0",
            },
        )

    def complete(
        self,
        messages: list[Message],
        *,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> CompletionResult:
        """Send a completion request to the Copilot API.

        Args:
            messages: Conversation messages.
            model: Model identifier (e.g. "claude-sonnet-4.6", "claude-haiku-4.5").
            temperature: Sampling temperature.
            max_tokens: Maximum response tokens.

        Returns:
            CompletionResult with the model's response.

        Raises:
            ProviderError: If the API call fails.
        """
        api_messages = [{"role": m.role, "content": m.content} for m in messages]

        try:
            response = self._client.chat.completions.create(
                model=model,
                messages=api_messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            raise ProviderError(
                f"Copilot API call failed: {exc}",
                provider="copilot",
            ) from exc

        choice = response.choices[0]
        content = choice.message.content or ""

        usage = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return CompletionResult(
            content=content,
            model=response.model,
            usage=usage,
        )
