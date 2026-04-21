"""State definitions for the Dirigent orchestration graph."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Annotated

from langgraph.graph.message import add_messages


class PRStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    REJECTED = "rejected"
    APPROVED = "approved"


class DeveloperStatus(Enum):
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


@dataclass
class SubTask:
    """A single PR-sized unit of work decomposed by the Architect."""

    id: str
    title: str
    description: str
    branch_name: str
    status: PRStatus = PRStatus.PENDING


@dataclass
class DeveloperResult:
    """Output from a single Developer agent's attempt at a sub-task."""

    developer_id: str
    sub_task_id: str
    branch_name: str
    status: DeveloperStatus = DeveloperStatus.RUNNING
    diff_stats: str = ""
    test_output: str = ""
    error: str = ""


@dataclass
class ReviewVerdict:
    """The Reviewer's evaluation of a single developer result."""

    developer_id: str
    passed_tests: bool
    architectural_alignment: int  # 1-5
    diff_size_score: int  # 1-5 (smaller = better)
    notes: str = ""


@dataclass
class ReviewResult:
    """Aggregated output from the Reviewer node."""

    verdicts: list[ReviewVerdict] = field(default_factory=list)
    selected_developer_id: str = ""
    recommendation: str = ""
    should_retry: bool = False


def merge_developer_results(
    existing: list[DeveloperResult], new: list[DeveloperResult]
) -> list[DeveloperResult]:
    """Custom reducer: merge developer results by developer_id."""
    by_id = {r.developer_id: r for r in existing}
    for result in new:
        by_id[result.developer_id] = result
    return list(by_id.values())


@dataclass
class GraphState:
    """Top-level state flowing through the Dirigent graph.

    Uses LangGraph's Annotated reducer pattern for fields that
    accumulate across parallel branches.
    """

    # Input
    objective: str = ""
    repo_path: str = ""

    # Integration branch
    feature_branch: str = ""

    # Architect output
    plan: list[SubTask] = field(default_factory=list)
    current_pr_index: int = 0

    # Fan-out / Fan-in
    developer_results: Annotated[list[DeveloperResult], merge_developer_results] = field(
        default_factory=list
    )

    # Reviewer output
    review: ReviewResult = field(default_factory=ReviewResult)

    # Control flow
    human_approved: bool = False
    iteration: int = 0
    messages: Annotated[list, add_messages] = field(default_factory=list)
