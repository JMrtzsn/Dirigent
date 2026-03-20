"""Architect node — decomposes the objective into PR-sized sub-tasks.

Calls the LLM with repo context to produce a structured plan.
Falls back to stub plan if no LLM config is available (for testing).
"""

# NOTE: Do NOT use `from __future__ import annotations` in node files.
# LangGraph introspects the `config: RunnableConfig` parameter type at runtime.
# With PEP 563 (future annotations), the type becomes a string and LangGraph
# can't recognize it, so it won't inject the RunnableConfig.

import json
import logging
import re

from langchain_core.runnables import RunnableConfig

from dirigent.llm.config import Config, Role
from dirigent.llm.provider import Message, ProviderError
from dirigent.state import GraphState, PRStatus, SubTask
from dirigent.utils.repo import build_repo_context

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a Lead Architect. Your job is to decompose a software engineering \
objective into 2-5 small, atomic, reviewable pull requests.

Each PR should follow the "refactoring principle of small changes":
- One logical change per PR
- Easy to review (< 300 lines diff ideally)
- Independently testable
- Ordered by dependency (earlier PRs don't depend on later ones)

You will receive:
1. The objective (what the user wants done)
2. The repository file tree and key config files

Respond with ONLY a JSON array of sub-tasks. No markdown, no explanation. Example:

[
  {
    "id": "pr-1",
    "title": "Extract auth interfaces",
    "description": "Define AuthProvider and AuthToken interfaces in auth/types.py",
    "branch_name": "feat/pr-1-auth-interfaces"
  },
  {
    "id": "pr-2",
    "title": "Implement JWT provider",
    "description": "Add JWTAuthProvider implementing AuthProvider interface",
    "branch_name": "feat/pr-2-jwt-provider"
  }
]

Rules:
- IDs must be sequential: pr-1, pr-2, pr-3, etc.
- Branch names must be valid git branch names (no spaces, lowercase, use hyphens)
- Descriptions should be specific enough for a developer to implement without ambiguity
- Order by dependency: foundational changes first
"""


def _get_config(runnable_config: RunnableConfig | None) -> Config | None:
    """Extract Dirigent Config from LangGraph's RunnableConfig."""
    if runnable_config is None:
        return None
    configurable = runnable_config.get("configurable", {})
    return configurable.get("dirigent_config")


def architect_node(state: GraphState, config: RunnableConfig | None = None) -> dict:
    """Analyze the objective and decompose into atomic, reviewable PRs.

    If a Dirigent Config with an LLM provider is available (via RunnableConfig),
    calls the LLM. Otherwise falls back to stub output.

    Args:
        state: Current graph state.
        config: LangGraph runnable config containing dirigent_config.

    Returns:
        Updated state with a plan of sub-tasks.
    """
    # If we already have a plan, this is a retry — don't regenerate
    if state.plan:
        logger.info("Plan already exists, skipping architect (iteration %d)", state.iteration)
        return {}

    logger.info("Architect decomposing objective: %s", state.objective)

    dirigent_config = _get_config(config)
    if dirigent_config is None:
        logger.info("No LLM config available, using stub plan")
        plan = _stub_plan(state.objective)
    else:
        plan = _llm_plan(state.objective, state.repo_path, dirigent_config)

    logger.info("Architect produced %d sub-tasks", len(plan))
    return {"plan": plan}


def _llm_plan(objective: str, repo_path: str, config: Config) -> list[SubTask]:
    """Generate a plan using the LLM provider."""
    repo_context = build_repo_context(repo_path)

    user_prompt = f"## Objective\n{objective}\n\n## Repository Context\n{repo_context}"

    model_config = config.model_for(Role.ARCHITECT)
    messages = [
        Message(role="system", content=_SYSTEM_PROMPT),
        Message(role="user", content=user_prompt),
    ]

    try:
        result = config.provider.complete(
            messages,
            model=model_config.model,
            temperature=model_config.temperature,
            max_tokens=model_config.max_tokens,
        )
    except ProviderError:
        logger.exception("LLM call failed, falling back to stub plan")
        return _stub_plan(objective)

    return _parse_plan(result.content, objective)


def _parse_plan(raw_response: str, objective: str) -> list[SubTask]:
    """Parse the LLM's JSON response into a list of SubTask objects.

    Handles common LLM quirks: markdown code fences, trailing commas, etc.
    Falls back to stub plan if parsing fails.
    """
    # Strip markdown code fences if present
    cleaned = raw_response.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()

    try:
        items = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("Failed to parse LLM response as JSON, falling back to stub plan")
        logger.debug("Raw response: %s", raw_response)
        return _stub_plan(objective)

    if not isinstance(items, list) or len(items) == 0:
        logger.warning("LLM response is not a non-empty list, falling back to stub plan")
        return _stub_plan(objective)

    plan: list[SubTask] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            plan.append(
                SubTask(
                    id=str(item["id"]),
                    title=str(item["title"]),
                    description=str(item["description"]),
                    branch_name=str(item["branch_name"]),
                    status=PRStatus.PENDING,
                )
            )
        except KeyError as exc:
            logger.warning("Skipping malformed sub-task (missing key %s): %s", exc, item)

    if not plan:
        logger.warning("No valid sub-tasks parsed, falling back to stub plan")
        return _stub_plan(objective)

    return plan


def _stub_plan(objective: str) -> list[SubTask]:
    """Placeholder plan generation for testing without an LLM."""
    return [
        SubTask(
            id="pr-1",
            title="Extract interfaces",
            description=f"Extract interfaces from existing code for: {objective}",
            branch_name="feat/pr-1-interfaces",
            status=PRStatus.PENDING,
        ),
        SubTask(
            id="pr-2",
            title="Implement core logic",
            description=f"Implement core logic for: {objective}",
            branch_name="feat/pr-2-core",
            status=PRStatus.PENDING,
        ),
        SubTask(
            id="pr-3",
            title="Add tests and integration",
            description=f"Add tests and wire up integration for: {objective}",
            branch_name="feat/pr-3-tests",
            status=PRStatus.PENDING,
        ),
    ]
