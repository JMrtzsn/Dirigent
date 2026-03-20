"""Developer node — executes a sub-task in an isolated git branch.

Each developer instance receives its work via LangGraph's Send API,
runs in parallel with other developers, and returns its result.
In the real implementation, this will:
1. Create a git worktree for isolation
2. Call an LLM (or CLI tool) to generate code
3. Run tests
4. Report results back
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from dirigent.state import DeveloperResult, DeveloperStatus, SubTask

logger = logging.getLogger(__name__)


@dataclass
class DeveloperInput:
    """Input passed to each developer via Send."""

    developer_id: str
    sub_task: SubTask
    branch_name: str
    repo_path: str


def developer_node(state: DeveloperInput | dict) -> dict:
    """Execute a sub-task and return the result.

    This node is invoked via Send — it receives DeveloperInput, not the full GraphState.
    LangGraph runs multiple instances of this node in parallel during fan-out.

    Returns:
        State update with developer_results containing this developer's output.
    """
    # Handle both dataclass and dict input from Send
    if isinstance(state, dict):
        developer_id = state["developer_id"]
        sub_task = state["sub_task"]
        branch_name = state["branch_name"]
    else:
        developer_id = state.developer_id
        sub_task = state.sub_task
        branch_name = state.branch_name

    logger.info(
        "Developer %s starting work on '%s' (branch: %s)",
        developer_id,
        sub_task.title,
        branch_name,
    )

    # STUB: Replace with actual implementation
    # 1. git worktree add <path> -b <branch_name>
    # 2. invoke LLM / CLI tool to generate code in worktree
    # 3. run tests in worktree
    # 4. collect diff stats
    result = _stub_developer_work(developer_id, sub_task, branch_name)

    logger.info("Developer %s finished with status: %s", developer_id, result.status.value)

    return {"developer_results": [result]}


def _stub_developer_work(
    developer_id: str, sub_task: SubTask, branch_name: str
) -> DeveloperResult:
    """Placeholder developer work. Replace with real code generation."""
    return DeveloperResult(
        developer_id=developer_id,
        sub_task_id=sub_task.id,
        branch_name=branch_name,
        status=DeveloperStatus.SUCCESS,
        diff_stats=f"+42 -17 across 3 files (stub from {developer_id})",
        test_output="All 12 tests passed (stub)",
        error="",
    )
