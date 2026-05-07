"""Dirigent orchestrator — SLIB workflow built on the agent interaction framework.

Replaces the LangGraph-based graph.py with a straightforward async pipeline
using the new core primitives: Runtime, LLMAgent, patterns.

Workflow:
    1. Architect decomposes objective into sub-tasks
    2. For each sub-task:
       a. Fan out to N developer agents
       b. Reviewer scores and selects the best
       c. Human approves (or rejects for retry)
       d. Winner is merged into the feature branch
    3. Push feature branch and open draft PR
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dirigent.core.message import DelegateRequest, DelegateResponse
from dirigent.llm.agent import LLMAgent
from dirigent.llm.copilot_provider import CopilotLLMProvider
from dirigent.llm.provider import LLMResponse
from dirigent.observe.metrics import Metrics
from dirigent.patterns.fanout import fanout
from dirigent.runtime import Runtime
from dirigent.utils.git import (
    GitError,
    create_draft_pr,
    create_feature_branch,
    delete_branch,
    merge_branch,
)
from dirigent.utils.repo import build_repo_context
from dirigent.utils.worktree import WorktreeManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class SubTask:
    """A single PR-sized unit of work."""

    id: str
    title: str
    description: str
    branch_name: str


@dataclass
class DeveloperOutput:
    """Result from one developer's attempt."""

    developer_id: str
    branch_name: str
    success: bool
    diff_stats: str = ""
    test_output: str = ""
    error: str = ""


@dataclass
class OrchestratorConfig:
    """Configuration for the orchestrator pipeline."""

    repo_path: str
    objective: str
    num_developers: int = 3
    max_review_retries: int = 2
    architect_model: str = "claude-sonnet-4.6"
    developer_model: str = "claude-haiku-4.5"
    reviewer_model: str = "claude-sonnet-4.6"
    human_approve_callback: Any = None  # async callable(summary) -> bool


# ---------------------------------------------------------------------------
# Agent definitions
# ---------------------------------------------------------------------------

_ARCHITECT_SYSTEM = """\
You are a Lead Architect. Decompose a software engineering objective into 2-5 \
small, atomic, reviewable pull requests.

Respond with ONLY a JSON array. Each item has keys: id, title, description, branch_name.
IDs are sequential (pr-1, pr-2, ...). Branch names are valid git branch names.
Order by dependency: foundational changes first.
"""

_DEVELOPER_SYSTEM = """\
You are a Software Developer. Implement a sub-task as file operations.

Respond with ONLY a JSON array of file operations:
[{"action": "create"|"overwrite"|"delete", "path": "relative/path", "content": "..."}]

Write complete file contents, not diffs. Production-quality code with type hints.
"""

_REVIEWER_SYSTEM = """\
You are a Code Reviewer. Evaluate developer implementations and select the best.

Respond with ONLY a JSON object:
{
  "selected": "developer_id",
  "reason": "why this one is best",
  "all_failed": false
}

Prefer implementations that pass tests and have small, focused diffs.
"""


class ArchitectAgent(LLMAgent):
    """Decomposes objectives into sub-tasks."""

    def build_messages(self, request: DelegateRequest) -> list[dict[str, str]]:
        messages = [{"role": "system", "content": _ARCHITECT_SYSTEM}]
        repo_context = request.context.get("repo_context", "")
        messages.append({
            "role": "user",
            "content": f"## Objective\n{request.task}\n\n## Repository\n{repo_context}",
        })
        return messages

    def parse_response(self, llm_response: LLMResponse, request: DelegateRequest) -> Any:
        return _parse_json_array(llm_response.content)


class DeveloperAgent(LLMAgent):
    """Implements a sub-task in an isolated worktree."""

    def build_messages(self, request: DelegateRequest) -> list[dict[str, str]]:
        messages = [{"role": "system", "content": _DEVELOPER_SYSTEM}]
        sub_task = request.context.get("sub_task", {})
        repo_context = request.context.get("repo_context", "")
        messages.append({
            "role": "user",
            "content": (
                f"## Sub-Task\n"
                f"**Title:** {sub_task.get('title', '')}\n"
                f"**Description:** {sub_task.get('description', '')}\n\n"
                f"## Repository\n{repo_context}"
            ),
        })
        return messages

    def parse_response(self, llm_response: LLMResponse, request: DelegateRequest) -> Any:
        return _parse_json_array(llm_response.content)


class ReviewerAgent(LLMAgent):
    """Evaluates developer outputs and picks the best."""

    def build_messages(self, request: DelegateRequest) -> list[dict[str, str]]:
        messages = [{"role": "system", "content": _REVIEWER_SYSTEM}]
        messages.append({"role": "user", "content": request.task})
        return messages

    def parse_response(self, llm_response: LLMResponse, request: DelegateRequest) -> Any:
        return _parse_json_object(llm_response.content)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class Orchestrator:
    """Runs the full SLIB workflow using the agent framework."""

    def __init__(self, config: OrchestratorConfig) -> None:
        self.config = config
        self.metrics = Metrics()
        self._runtime = Runtime()
        self._provider = CopilotLLMProvider()
        self._setup_agents()

    def _setup_agents(self) -> None:
        architect = ArchitectAgent(
            name="architect",
            capabilities=["decompose"],
            provider=self._provider,
            model=self.config.architect_model,
        )

        # Register N developer agents
        for i in range(self.config.num_developers):
            dev = DeveloperAgent(
                name=f"dev-{i}",
                capabilities=["implement"],
                provider=self._provider,
                model=self.config.developer_model,
            )
            self._runtime.register(dev)

        reviewer = ReviewerAgent(
            name="reviewer",
            capabilities=["review"],
            provider=self._provider,
            model=self.config.reviewer_model,
        )

        self._runtime.register(architect, reviewer)

    async def run(self) -> str | None:
        """Execute the full pipeline. Returns the draft PR URL or None."""
        repo_path = Path(self.config.repo_path)
        repo_context = build_repo_context(self.config.repo_path)

        # 1. Architect decomposes
        logger.info("Phase: ARCHITECT")
        plan_response = await self._runtime.delegate(
            to="architect",
            task=self.config.objective,
            context={"repo_context": repo_context},
        )

        if not plan_response.success:
            logger.error("Architect failed: %s", plan_response.error)
            return None

        plan = _build_plan(plan_response.result)
        if not plan:
            logger.error("Architect produced empty plan")
            return None

        logger.info("Architect produced %d sub-tasks", len(plan))

        # 2. Create feature branch
        feature_branch = create_feature_branch(repo_path, self.config.objective)
        logger.info("Feature branch: %s", feature_branch)

        # 3. For each sub-task
        for task_idx, sub_task in enumerate(plan):
            logger.info("Sub-task %d/%d: %s", task_idx + 1, len(plan), sub_task.title)

            winner = await self._execute_subtask(sub_task, repo_path, feature_branch, repo_context)
            if winner is None:
                logger.error("Sub-task failed after retries, aborting")
                return None

            # Merge winner
            try:
                merge_branch(repo_path, winner.branch_name, feature_branch)
            except GitError:
                logger.exception("Failed to merge %s", winner.branch_name)
                return None

            # Cleanup dev branches
            for i in range(self.config.num_developers):
                delete_branch(repo_path, f"{sub_task.branch_name}/dev-{i}")

        # 4. Open draft PR
        try:
            pr_url = create_draft_pr(repo_path, feature_branch, self.config.objective)
            logger.info("Draft PR: %s", pr_url)
            return pr_url
        except GitError:
            logger.exception("Failed to create draft PR")
            return None

    async def _execute_subtask(
        self,
        sub_task: SubTask,
        repo_path: Path,
        feature_branch: str,
        repo_context: str,
    ) -> DeveloperOutput | None:
        """Fan out to developers, review, get human approval. Retries on rejection."""
        for attempt in range(self.config.max_review_retries + 1):
            # Fan out to all developers
            dev_names = [f"dev-{i}" for i in range(self.config.num_developers)]
            dev_results = await fanout(
                self._runtime,
                agents=dev_names,
                task=f"Implement: {sub_task.title}",
                context={
                    "sub_task": {
                        "title": sub_task.title,
                        "description": sub_task.description,
                    },
                    "repo_context": repo_context,
                },
            )

            # Apply file ops in worktrees and collect outputs
            outputs = await self._apply_in_worktrees(
                dev_results, sub_task, repo_path, feature_branch
            )

            successful = [o for o in outputs if o.success]
            if not successful:
                logger.warning("All developers failed on attempt %d", attempt + 1)
                continue

            # Review
            review_summary = _format_review_input(outputs)
            review_response = await self._runtime.delegate(
                to="reviewer",
                task=review_summary,
            )

            if not review_response.success:
                logger.warning("Reviewer failed, picking first successful dev")
                return successful[0]

            selected_id = _extract_winner(review_response.result, successful)
            winner = next((o for o in successful if o.developer_id == selected_id), successful[0])

            # Human approval
            if self.config.human_approve_callback:
                summary = f"Reviewer selected: {winner.developer_id}\nDiff: {winner.diff_stats}"
                approved = await self.config.human_approve_callback(summary)
                if not approved:
                    logger.info("Human rejected, retrying (attempt %d)", attempt + 1)
                    continue

            return winner

        return None

    async def _apply_in_worktrees(
        self,
        dev_results: list[DelegateResponse],
        sub_task: SubTask,
        repo_path: Path,
        feature_branch: str,
    ) -> list[DeveloperOutput]:
        """Apply each developer's file operations in isolated worktrees."""
        manager = WorktreeManager(repo_path)
        outputs: list[DeveloperOutput] = []

        for i, resp in enumerate(dev_results):
            dev_id = f"dev-{i}"
            branch_name = f"{sub_task.branch_name}/{dev_id}"

            if not resp.success or not resp.result:
                outputs.append(DeveloperOutput(
                    developer_id=dev_id,
                    branch_name=branch_name,
                    success=False,
                    error=resp.error or "No file operations produced",
                ))
                continue

            try:
                worktree = manager.create(branch_name, dev_id, start_point=feature_branch)
                try:
                    _apply_file_ops(worktree.path, resp.result)
                    worktree.run_in_worktree(["git", "add", "-A"])
                    worktree.run_in_worktree([
                        "git", "commit", "-m", sub_task.title, "--allow-empty"
                    ])
                    stat = worktree.run_in_worktree(["git", "diff", "--stat", "HEAD~1..HEAD"])
                    diff_stats = stat.stdout.strip() if stat.returncode == 0 else ""

                    outputs.append(DeveloperOutput(
                        developer_id=dev_id,
                        branch_name=branch_name,
                        success=True,
                        diff_stats=diff_stats,
                    ))
                finally:
                    manager.remove(worktree)
            except Exception as exc:
                logger.warning("Worktree failed for %s: %s", dev_id, exc)
                outputs.append(DeveloperOutput(
                    developer_id=dev_id,
                    branch_name=branch_name,
                    success=False,
                    error=str(exc),
                ))

        return outputs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_json_array(raw: str) -> list[dict[str, Any]]:
    """Parse a JSON array from LLM output, stripping code fences."""
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    try:
        result = json.loads(cleaned)
        return result if isinstance(result, list) else []
    except json.JSONDecodeError:
        return []


def _parse_json_object(raw: str) -> dict[str, Any]:
    """Parse a JSON object from LLM output."""
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    try:
        result = json.loads(cleaned)
        return result if isinstance(result, dict) else {}
    except json.JSONDecodeError:
        return {}


def _build_plan(raw_items: list[dict[str, Any]] | Any) -> list[SubTask]:
    """Convert raw JSON items into SubTask objects."""
    if not isinstance(raw_items, list):
        return []
    tasks: list[SubTask] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        try:
            tasks.append(SubTask(
                id=str(item["id"]),
                title=str(item["title"]),
                description=str(item["description"]),
                branch_name=str(item["branch_name"]),
            ))
        except KeyError:
            continue
    return tasks


def _apply_file_ops(worktree_path: Path, ops: list[dict[str, Any]]) -> None:
    """Apply file operations to a worktree directory."""
    for op in ops:
        if not isinstance(op, dict):
            continue
        action = op.get("action", "")
        path = op.get("path", "")
        if not path:
            continue
        target = worktree_path / path
        if action == "delete":
            if target.exists():
                target.unlink()
        elif action in ("create", "overwrite"):
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(op.get("content", ""), encoding="utf-8")


def _format_review_input(outputs: list[DeveloperOutput]) -> str:
    """Format developer outputs for the reviewer."""
    parts: list[str] = []
    for o in outputs:
        status = "SUCCESS" if o.success else "FAILED"
        parts.append(
            f"### {o.developer_id} ({status})\n"
            f"Diff: {o.diff_stats or '(none)'}\n"
            f"Tests: {o.test_output or '(none)'}\n"
            f"Error: {o.error or '(none)'}\n"
        )
    return "## Developer Results\n\n" + "\n".join(parts)


def _extract_winner(review_result: dict[str, Any] | Any, successful: list[DeveloperOutput]) -> str:
    """Extract the selected developer ID from the reviewer's response."""
    if isinstance(review_result, dict):
        selected = review_result.get("selected", "")
        if selected:
            return selected
    # Fallback: first successful
    return successful[0].developer_id if successful else ""
