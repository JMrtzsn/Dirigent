"""Dirigent graph definition — the core orchestration topology.

SLIB (Short-Lived Integration Branch) workflow:
    Architect → Create Feature Branch → [Fan Out via Send] → Developer × N
    → [Fan In] → Reviewer → Human Review → Merge Winner → next PR or Draft PR → END

Each sub-task is sequential: the next one branches off the feature branch
after the previous winner is merged in.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from langgraph.graph import END, StateGraph
from langgraph.types import Send, interrupt

from dirigent.nodes.architect import architect_node
from dirigent.nodes.developer import developer_node
from dirigent.nodes.reviewer import reviewer_node
from dirigent.state import GraphState, ReviewResult
from dirigent.utils.git import (
    GitError,
    create_draft_pr,
    create_feature_branch,
    delete_branch,
    merge_branch,
)

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver

logger = logging.getLogger(__name__)

NUM_DEVELOPERS = 3


def setup_feature_branch(state: GraphState) -> dict:
    """Create the integration branch after the architect produces a plan."""
    if state.feature_branch:
        return {}

    repo_path = Path(state.repo_path)
    branch = create_feature_branch(repo_path, state.objective)
    return {"feature_branch": branch}


def fan_out_to_developers(state: GraphState) -> list[Send]:
    """Dynamic fan-out: spawn N developer agents for the current sub-task.

    Each developer branches off the feature branch so it includes
    all previously merged work.
    """
    current_task = state.plan[state.current_pr_index]
    sends = []
    for i in range(NUM_DEVELOPERS):
        developer_id = f"dev-{i}"
        branch = f"{current_task.branch_name}/{developer_id}"
        sends.append(
            Send(
                "developer",
                {
                    "developer_id": developer_id,
                    "sub_task": current_task,
                    "branch_name": branch,
                    "repo_path": state.repo_path,
                    "feature_branch": state.feature_branch,
                },
            )
        )
    return sends


def human_review(state: GraphState) -> GraphState:
    """Human-in-the-loop gate. Pauses execution for manual approval."""
    review = state.review

    summary = (
        f"Reviewer recommends: {review.selected_developer_id}\n"
        f"Reason: {review.recommendation}\n\n"
        "Approve this selection? (interrupt will pause here)"
    )

    decision = interrupt(summary)

    approved = decision.lower() in ("y", "yes", "approve")
    return {"human_approved": approved}


def merge_winner(state: GraphState) -> dict:
    """Merge the winning developer's branch into the feature branch.

    Also cleans up losing developer branches.
    """
    repo_path = Path(state.repo_path)
    winner_id = state.review.selected_developer_id
    current_task = state.plan[state.current_pr_index]

    winner_branch = f"{current_task.branch_name}/{winner_id}"

    try:
        merge_branch(repo_path, winner_branch, state.feature_branch)
    except GitError:
        logger.exception("Failed to merge winner branch %s", winner_branch)
        raise

    # Clean up all developer branches for this sub-task
    for i in range(NUM_DEVELOPERS):
        dev_branch = f"{current_task.branch_name}/dev-{i}"
        delete_branch(repo_path, dev_branch)

    return {}


def should_continue_after_review(state: GraphState) -> str:
    """Conditional edge: retry fan-out or proceed to human gate."""
    if state.review.should_retry:
        return "fan_out"
    return "human_review"


def should_continue_after_human(state: GraphState) -> str:
    """Conditional edge: reject (retry) or proceed to merge."""
    if not state.human_approved:
        return "fan_out"
    return "merge_winner"


def should_continue_after_merge(state: GraphState) -> str:
    """After merging, advance to next PR or finalize."""
    next_index = state.current_pr_index + 1
    if next_index < len(state.plan):
        return "advance_pr"
    return "finalize"


def advance_pr(state: GraphState) -> GraphState:
    """Move to the next PR in the plan and reset per-PR state."""
    return GraphState(
        current_pr_index=state.current_pr_index + 1,
        developer_results=[],
        review=ReviewResult(),
        human_approved=False,
        iteration=state.iteration + 1,
    )


def finalize(state: GraphState) -> dict:
    """Push the feature branch and create a draft PR."""
    repo_path = Path(state.repo_path)
    try:
        pr_url = create_draft_pr(repo_path, state.feature_branch, state.objective)
        logger.info("Draft PR created: %s", pr_url)
    except GitError:
        logger.exception("Failed to create draft PR — feature branch is still available locally")
    return {}


def build_graph(*, checkpointer: BaseCheckpointSaver | None = None) -> StateGraph:
    """Construct the Dirigent orchestration graph.

    Args:
        checkpointer: Optional checkpoint saver for persistent state.
            Required for human-in-the-loop interrupt() to survive restarts.

    Returns a compiled StateGraph ready for execution.
    """
    builder = StateGraph(GraphState)

    # Nodes
    builder.add_node("architect", architect_node)
    builder.add_node("setup_feature_branch", setup_feature_branch)
    builder.add_node("developer", developer_node)
    builder.add_node("reviewer", reviewer_node)
    builder.add_node("human_review", human_review)
    builder.add_node("merge_winner", merge_winner)
    builder.add_node("advance_pr", advance_pr)
    builder.add_node("finalize", finalize)

    # Edges
    builder.set_entry_point("architect")

    # Architect → setup feature branch
    builder.add_edge("architect", "setup_feature_branch")

    # Feature branch → Fan Out (dynamic edge using Send)
    builder.add_conditional_edges("setup_feature_branch", fan_out_to_developers)

    # Fan In: all developer Send nodes converge to reviewer
    builder.add_edge("developer", "reviewer")

    # Reviewer → conditionally retry or go to human gate
    builder.add_conditional_edges(
        "reviewer",
        should_continue_after_review,
        {"fan_out": "setup_feature_branch", "human_review": "human_review"},
    )

    # Human gate → merge or retry
    builder.add_conditional_edges(
        "human_review",
        should_continue_after_human,
        {"fan_out": "setup_feature_branch", "merge_winner": "merge_winner"},
    )

    # After merge → advance to next PR or finalize
    builder.add_conditional_edges(
        "merge_winner",
        should_continue_after_merge,
        {"advance_pr": "advance_pr", "finalize": "finalize"},
    )

    # After advancing PR, fan out again for next task
    builder.add_conditional_edges("advance_pr", fan_out_to_developers)

    # Finalize → END
    builder.add_edge("finalize", END)

    return builder.compile(checkpointer=checkpointer)
