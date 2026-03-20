"""Developer node — executes a sub-task in an isolated git worktree.

Each developer instance receives its work via LangGraph's Send API,
runs in parallel with other developers, and returns its result.

Workflow:
1. Create a git worktree for isolation
2. Call the LLM to generate code changes (as JSON file operations)
3. Apply file operations to the worktree
4. Commit changes and collect diff stats
5. Optionally run tests
6. Return DeveloperResult

Falls back to stub output if no LLM config is available.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from langchain_core.runnables import RunnableConfig

from dirigent.llm.config import Config, Role
from dirigent.llm.provider import Message, ProviderError
from dirigent.state import DeveloperResult, DeveloperStatus, SubTask
from dirigent.utils.repo import build_repo_context
from dirigent.utils.worktree import Worktree, WorktreeError, WorktreeManager

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a Software Developer. You receive a sub-task describing a single, \
small pull request to implement. You write clean, idiomatic code.

You will receive:
1. The sub-task (title, description, branch context)
2. The repository file tree and key config files for context

Respond with ONLY a JSON array of file operations. No markdown, no explanation.

Each operation is an object with:
- "action": one of "create", "overwrite", "delete"
- "path": relative file path from repo root (e.g. "src/auth/types.py")
- "content": file content (required for "create" and "overwrite", omit for "delete")

Example:
[
  {
    "action": "create",
    "path": "src/auth/types.py",
    "content": "from __future__ import annotations\\n\\nclass AuthToken:\\n    ..."
  },
  {
    "action": "overwrite",
    "path": "src/auth/__init__.py",
    "content": "from .types import AuthToken\\n"
  }
]

Rules:
- Output complete file contents, not diffs or patches
- Write production-quality code with type hints and docstrings
- Follow existing code style and conventions visible in the repo context
- Only touch files directly relevant to the sub-task
- Keep changes minimal and focused
"""


def _get_config(runnable_config: RunnableConfig | None) -> Config | None:
    """Extract Dirigent Config from LangGraph's RunnableConfig."""
    if runnable_config is None:
        return None
    configurable = runnable_config.get("configurable", {})
    return configurable.get("dirigent_config")


@dataclass
class DeveloperInput:
    """Input passed to each developer via Send."""

    developer_id: str
    sub_task: SubTask
    branch_name: str
    repo_path: str


def developer_node(state: DeveloperInput | dict, config: RunnableConfig | None = None) -> dict:
    """Execute a sub-task and return the result.

    This node is invoked via Send — it receives DeveloperInput, not the full GraphState.
    LangGraph runs multiple instances of this node in parallel during fan-out.

    If a Dirigent Config with an LLM provider is available (via RunnableConfig),
    creates a worktree, calls the LLM, and applies changes. Otherwise uses stub output.

    Args:
        state: Developer input (task details).
        config: LangGraph runnable config containing dirigent_config.

    Returns:
        State update with developer_results containing this developer's output.
    """
    if isinstance(state, dict):
        developer_id = state["developer_id"]
        sub_task = state["sub_task"]
        branch_name = state["branch_name"]
        repo_path = state.get("repo_path", "")
    else:
        developer_id = state.developer_id
        sub_task = state.sub_task
        branch_name = state.branch_name
        repo_path = state.repo_path

    logger.info(
        "Developer %s starting work on '%s' (branch: %s)",
        developer_id,
        sub_task.title,
        branch_name,
    )

    dirigent_config = _get_config(config)
    if dirigent_config is None or not repo_path:
        logger.info("Developer %s: no LLM config or repo path, using stub", developer_id)
        result = _stub_developer_work(developer_id, sub_task, branch_name)
    else:
        result = _llm_developer_work(
            developer_id, sub_task, branch_name, repo_path, dirigent_config
        )

    logger.info("Developer %s finished with status: %s", developer_id, result.status.value)
    return {"developer_results": [result]}


def _llm_developer_work(
    developer_id: str,
    sub_task: SubTask,
    branch_name: str,
    repo_path: str,
    config: Config,
) -> DeveloperResult:
    """Execute a sub-task using LLM + git worktree."""
    manager = WorktreeManager(Path(repo_path))
    worktree: Worktree | None = None

    try:
        worktree = manager.create(branch_name, developer_id)
        file_ops = _call_llm_for_code(sub_task, repo_path, config)
        _apply_file_operations(worktree, file_ops)
        diff_stats = _commit_changes(worktree, sub_task)
        test_output = _run_tests(worktree)

        return DeveloperResult(
            developer_id=developer_id,
            sub_task_id=sub_task.id,
            branch_name=branch_name,
            status=DeveloperStatus.SUCCESS,
            diff_stats=diff_stats,
            test_output=test_output,
        )
    except (WorktreeError, ProviderError) as exc:
        logger.exception("Developer %s failed", developer_id)
        return DeveloperResult(
            developer_id=developer_id,
            sub_task_id=sub_task.id,
            branch_name=branch_name,
            status=DeveloperStatus.FAILED,
            error=str(exc),
        )
    finally:
        if worktree is not None:
            try:
                manager.remove(worktree)
            except Exception:
                logger.warning("Failed to clean up worktree for %s", developer_id)


@dataclass
class FileOperation:
    """A single file operation from the LLM response."""

    action: str  # "create", "overwrite", "delete"
    path: str
    content: str = ""


def _call_llm_for_code(sub_task: SubTask, repo_path: str, config: Config) -> list[FileOperation]:
    """Call the LLM to generate code changes for a sub-task."""
    repo_context = build_repo_context(repo_path)

    user_prompt = (
        f"## Sub-Task\n"
        f"**Title:** {sub_task.title}\n"
        f"**Description:** {sub_task.description}\n"
        f"**Branch:** {sub_task.branch_name}\n\n"
        f"## Repository Context\n{repo_context}"
    )

    model_config = config.model_for(Role.DEVELOPER)
    messages = [
        Message(role="system", content=_SYSTEM_PROMPT),
        Message(role="user", content=user_prompt),
    ]

    result = config.provider.complete(
        messages,
        model=model_config.model,
        temperature=model_config.temperature,
        max_tokens=model_config.max_tokens,
    )

    return _parse_file_operations(result.content)


def _parse_file_operations(raw_response: str) -> list[FileOperation]:
    """Parse the LLM's JSON response into file operations.

    Handles markdown code fences. Raises ValueError if parsing fails completely.
    """
    cleaned = raw_response.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()

    try:
        items = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        msg = f"Failed to parse LLM response as JSON: {exc}"
        raise ProviderError(msg, provider="developer-parser") from exc

    if not isinstance(items, list):
        msg = "LLM response is not a JSON array"
        raise ProviderError(msg, provider="developer-parser")

    ops: list[FileOperation] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        action = item.get("action", "")
        path = item.get("path", "")
        if action not in ("create", "overwrite", "delete") or not path:
            logger.warning("Skipping invalid file operation: %s", item)
            continue
        ops.append(
            FileOperation(
                action=action,
                path=path,
                content=item.get("content", ""),
            )
        )

    if not ops:
        msg = "No valid file operations parsed from LLM response"
        raise ProviderError(msg, provider="developer-parser")

    return ops


def _apply_file_operations(worktree: Worktree, ops: list[FileOperation]) -> None:
    """Write/delete files in the worktree according to the LLM's instructions."""
    for op in ops:
        target = worktree.path / op.path
        if op.action == "delete":
            if target.exists():
                target.unlink()
                logger.info("Deleted %s", op.path)
            else:
                logger.warning("Cannot delete non-existent file: %s", op.path)
        elif op.action in ("create", "overwrite"):
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(op.content, encoding="utf-8")
            logger.info("%s %s", op.action.capitalize(), op.path)


def _commit_changes(worktree: Worktree, sub_task: SubTask) -> str:
    """Stage all changes in the worktree and commit. Returns diff stats."""
    worktree.run_in_worktree(["git", "add", "-A"])

    commit_msg = f"{sub_task.title}\n\n{sub_task.description}"
    result = worktree.run_in_worktree(["git", "commit", "-m", commit_msg, "--allow-empty"])
    if result.returncode != 0:
        logger.warning("git commit failed: %s", result.stderr)

    stat_result = worktree.run_in_worktree(["git", "diff", "--stat", "HEAD~1..HEAD"])
    return stat_result.stdout.strip() if stat_result.returncode == 0 else "(no diff stats)"


def _run_tests(worktree: Worktree) -> str:
    """Attempt to run tests in the worktree. Best-effort, non-fatal."""
    test_commands = [
        (["make", "check"], "Makefile"),
        (["python", "-m", "pytest", "--tb=short", "-q"], "pytest"),
    ]

    for cmd, label in test_commands:
        check_file = worktree.path / cmd[0] if label == "Makefile" else None
        if check_file and label == "Makefile" and not (worktree.path / "Makefile").exists():
            continue

        result = worktree.run_in_worktree(cmd, timeout=120)
        output = result.stdout + result.stderr
        status = "passed" if result.returncode == 0 else "failed"
        logger.info("Tests (%s): %s (exit %d)", label, status, result.returncode)
        return f"[{label}] exit {result.returncode}\n{output}".strip()

    return "(no test runner detected)"


def _stub_developer_work(
    developer_id: str, sub_task: SubTask, branch_name: str
) -> DeveloperResult:
    """Placeholder developer work for testing without an LLM."""
    return DeveloperResult(
        developer_id=developer_id,
        sub_task_id=sub_task.id,
        branch_name=branch_name,
        status=DeveloperStatus.SUCCESS,
        diff_stats=f"+42 -17 across 3 files (stub from {developer_id})",
        test_output="All 12 tests passed (stub)",
        error="",
    )
