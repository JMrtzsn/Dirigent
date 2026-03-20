"""Architect node — decomposes the objective into PR-sized sub-tasks.

In the real implementation, this will call an LLM to analyze the codebase
and produce a plan. For now, it returns a stubbed plan.
"""

from __future__ import annotations

import logging

from dirigent.state import GraphState, PRStatus, SubTask

logger = logging.getLogger(__name__)


def architect_node(state: GraphState) -> dict:
    """Analyze the objective and decompose into atomic, reviewable PRs.

    Returns:
        Updated state with a plan of sub-tasks.
    """
    # If we already have a plan, this is a retry — don't regenerate
    if state.plan:
        logger.info("Plan already exists, skipping architect (iteration %d)", state.iteration)
        return {}

    logger.info("Architect decomposing objective: %s", state.objective)

    # STUB: Replace with LLM call that analyzes the repo and produces a plan
    plan = _stub_plan(state.objective)

    logger.info("Architect produced %d sub-tasks", len(plan))
    return {"plan": plan}


def _stub_plan(objective: str) -> list[SubTask]:
    """Placeholder plan generation. Replace with LLM-powered decomposition."""
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
