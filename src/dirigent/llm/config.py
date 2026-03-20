"""Role-based model configuration for Dirigent.

Maps each agent role (architect, developer, reviewer) to an LLM model and provider.
The architect and reviewer need stronger reasoning; developers can use a cheaper model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

from dirigent.llm.copilot import CopilotProvider
from dirigent.llm.provider import LLMProvider

logger = logging.getLogger(__name__)


class Role(Enum):
    """Agent roles in the Dirigent graph."""

    ARCHITECT = "architect"
    DEVELOPER = "developer"
    REVIEWER = "reviewer"


@dataclass(frozen=True)
class ModelConfig:
    """Model assignment for a single role."""

    model: str
    temperature: float = 0.0
    max_tokens: int = 4096


# Defaults: strong model for planning/review, cheap model for code generation
DEFAULT_MODELS: dict[Role, ModelConfig] = {
    Role.ARCHITECT: ModelConfig(model="claude-sonnet-4.6", max_tokens=4096),
    Role.DEVELOPER: ModelConfig(model="claude-haiku-4.5", max_tokens=8192),
    Role.REVIEWER: ModelConfig(model="claude-sonnet-4.6", max_tokens=4096),
}


class Config:
    """Dirigent runtime configuration.

    Holds the LLM provider and per-role model assignments.
    Constructed once at startup and passed into the graph.

    Usage:
        config = Config()  # Uses defaults (Copilot + standard models)
        provider = config.provider
        model_cfg = config.model_for(Role.ARCHITECT)
    """

    def __init__(
        self,
        *,
        provider: LLMProvider | None = None,
        models: dict[Role, ModelConfig] | None = None,
    ) -> None:
        """Initialize configuration.

        Args:
            provider: LLM provider instance. Defaults to CopilotProvider.
            models: Per-role model overrides. Missing roles fall back to defaults.
        """
        self.provider = provider or CopilotProvider()
        self._models = {**DEFAULT_MODELS, **(models or {})}
        logger.info(
            "Config initialized: provider=%s, models=%s",
            type(self.provider).__name__,
            {r.value: m.model for r, m in self._models.items()},
        )

    def model_for(self, role: Role) -> ModelConfig:
        """Get the model configuration for a given role.

        Args:
            role: The agent role.

        Returns:
            ModelConfig for that role.
        """
        return self._models[role]
