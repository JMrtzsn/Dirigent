"""Tests for the LLM provider abstraction and config."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dirigent.llm.config import Config, ModelConfig, Role
from dirigent.llm.copilot import (
    CopilotProvider,
    _resolve_token,
)
from dirigent.llm.provider import (
    CompletionResult,
    LLMProvider,
    Message,
    ProviderError,
)

# --- Provider protocol tests ---


class TestLLMProviderProtocol:
    def test_copilot_satisfies_protocol(self) -> None:
        """CopilotProvider must satisfy the LLMProvider protocol."""
        assert issubclass(CopilotProvider, LLMProvider)

    def test_custom_provider_satisfies_protocol(self) -> None:
        """Any class with a matching complete() method satisfies the protocol."""

        class FakeProvider:
            def complete(
                self,
                messages: list[Message],
                *,
                model: str,
                temperature: float = 0.0,
                max_tokens: int = 4096,
            ) -> CompletionResult:
                return CompletionResult(content="fake", model=model)

        assert isinstance(FakeProvider(), LLMProvider)


class TestProviderError:
    def test_stores_provider_and_status(self) -> None:
        err = ProviderError("boom", provider="copilot", status_code=429)
        assert str(err) == "boom"
        assert err.provider == "copilot"
        assert err.status_code == 429

    def test_status_code_defaults_to_none(self) -> None:
        err = ProviderError("fail", provider="test")
        assert err.status_code is None


class TestMessage:
    def test_construction(self) -> None:
        msg = Message(role="user", content="hello")
        assert msg.role == "user"
        assert msg.content == "hello"


class TestCompletionResult:
    def test_defaults(self) -> None:
        result = CompletionResult(content="hi", model="test-model")
        assert result.usage == {}

    def test_with_usage(self) -> None:
        result = CompletionResult(
            content="hi",
            model="test-model",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )
        assert result.usage["total_tokens"] == 15


# --- Token resolution tests ---


class TestResolveToken:
    def test_env_var_takes_priority(self, tmp_path: Path) -> None:
        """DIRIGENT_COPILOT_TOKEN env var should be used first."""
        with patch.dict("os.environ", {"DIRIGENT_COPILOT_TOKEN": "env-token-123"}):
            assert _resolve_token() == "env-token-123"

    def test_reads_opencode_auth_cache(self, tmp_path: Path) -> None:
        """Falls back to opencode's auth.json."""
        auth_file = tmp_path / "auth.json"
        auth_file.write_text(
            json.dumps(
                {
                    "github-copilot": {
                        "type": "oauth",
                        "access": "access-token",
                        "refresh": "refresh-token-456",
                        "expires": 0,
                    }
                }
            )
        )
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("dirigent.llm.copilot.OPENCODE_AUTH_PATH", auth_file),
        ):
            # Clear the env var explicitly
            os.environ.pop("DIRIGENT_COPILOT_TOKEN", None)
            assert _resolve_token() == "refresh-token-456"

    def test_raises_when_no_token_found(self, tmp_path: Path) -> None:
        """Raises ProviderError if no token source is available."""
        missing_path = tmp_path / "nonexistent" / "auth.json"
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("dirigent.llm.copilot.OPENCODE_AUTH_PATH", missing_path),
        ):
            os.environ.pop("DIRIGENT_COPILOT_TOKEN", None)
            with pytest.raises(ProviderError, match="No Copilot token found"):
                _resolve_token()

    def test_raises_on_malformed_auth_json(self, tmp_path: Path) -> None:
        """Handles corrupt auth.json gracefully."""
        auth_file = tmp_path / "auth.json"
        auth_file.write_text("not valid json{{{")
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("dirigent.llm.copilot.OPENCODE_AUTH_PATH", auth_file),
        ):
            os.environ.pop("DIRIGENT_COPILOT_TOKEN", None)
            with pytest.raises(ProviderError, match="No Copilot token found"):
                _resolve_token()


# --- CopilotProvider tests ---


class TestCopilotProvider:
    def test_constructs_with_explicit_token(self) -> None:
        """Should accept an explicit token without resolving."""
        provider = CopilotProvider(token="explicit-token")
        assert provider._client.base_url.host == "api.githubcopilot.com"

    def test_complete_calls_openai_client(self) -> None:
        """complete() should delegate to the OpenAI client."""
        provider = CopilotProvider(token="test-token")

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello from Copilot"
        mock_response.model = "claude-sonnet-4.6"
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15

        with patch.object(
            provider._client.chat.completions, "create", return_value=mock_response
        ) as mock_create:
            result = provider.complete(
                [Message(role="user", content="hi")],
                model="claude-sonnet-4.6",
            )

            mock_create.assert_called_once()
            call_kwargs = mock_create.call_args
            assert call_kwargs.kwargs["model"] == "claude-sonnet-4.6"
            assert result.content == "Hello from Copilot"
            assert result.usage["total_tokens"] == 15

    def test_complete_wraps_api_errors(self) -> None:
        """API failures should be wrapped in ProviderError."""
        provider = CopilotProvider(token="test-token")

        with (
            patch.object(
                provider._client.chat.completions,
                "create",
                side_effect=Exception("connection refused"),
            ),
            pytest.raises(ProviderError, match="Copilot API call failed"),
        ):
            provider.complete(
                [Message(role="user", content="hi")],
                model="claude-sonnet-4.6",
            )


# --- Config tests ---


class TestConfig:
    def test_default_models(self) -> None:
        """Default config uses standard model assignments."""
        config = Config(provider=MagicMock(spec=LLMProvider))
        assert config.model_for(Role.ARCHITECT).model == "claude-sonnet-4.6"
        assert config.model_for(Role.DEVELOPER).model == "claude-haiku-4.5"
        assert config.model_for(Role.REVIEWER).model == "claude-sonnet-4.6"

    def test_custom_model_override(self) -> None:
        """Custom models should override defaults for specified roles."""
        overrides = {
            Role.DEVELOPER: ModelConfig(model="gpt-4o-mini", max_tokens=16384),
        }
        config = Config(provider=MagicMock(spec=LLMProvider), models=overrides)
        assert config.model_for(Role.DEVELOPER).model == "gpt-4o-mini"
        assert config.model_for(Role.DEVELOPER).max_tokens == 16384
        # Non-overridden roles keep defaults
        assert config.model_for(Role.ARCHITECT).model == "claude-sonnet-4.6"

    def test_model_config_frozen(self) -> None:
        """ModelConfig should be immutable."""
        mc = ModelConfig(model="test")
        with pytest.raises(AttributeError):
            mc.model = "changed"
