"""Dirigent graph definition — the core orchestration topology.

Architecture:
    Architect → [Fan Out via Send] → Developer × N → [Fan In] → Reviewer → Human Stop
                                                                    ↓ (reject)
                                                               Back to Fan Out
"""

from __future__ import annotations

from langgraph.constants import Send
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt

from dirigent.nodes.architect import architect_node
from dirigent.nodes.developer import developer_node
from dirigent.nodes.reviewer import reviewer_node
from dirigent.state import GraphState, ReviewResult

NUM_DEVELOPERS = 3


def fan_out_to_developers(state: GraphState) -> list[Send]:
    """Dynamic fan-out: spawn N developer agents for the current sub-task.

    Each developer gets a unique ID and works on a dedicated branch.
    LangGraph's Send API runs these in parallel.
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

    # LangGraph interrupt — execution pauses, resumes when human responds
    decision = interrupt(summary)

    approved = decision.lower() in ("y", "yes", "approve")
    return GraphState(human_approved=approved)


def should_continue_after_review(state: GraphState) -> str:
    """Conditional edge: retry fan-out or proceed to human gate."""
    if state.review.should_retry:
        return "fan_out"
    return "human_review"


def should_continue_after_human(state: GraphState) -> str:
    """Conditional edge: advance to next PR or end."""
    if not state.human_approved:
        return "fan_out"

    next_index = state.current_pr_index + 1
    if next_index < len(state.plan):
        return "advance_pr"
    return END


def advance_pr(state: GraphState) -> GraphState:
    """Move to the next PR in the plan and reset per-PR state."""
    return GraphState(
        current_pr_index=state.current_pr_index + 1,
        developer_results=[],
        review=ReviewResult(),
        human_approved=False,
        iteration=state.iteration + 1,
    )


def build_graph() -> StateGraph:
    """Construct the Dirigent orchestration graph.

    Returns a compiled StateGraph ready for execution.
    """
    builder = StateGraph(GraphState)

    # Nodes
    builder.add_node("architect", architect_node)
    builder.add_node("developer", developer_node)
    builder.add_node("reviewer", reviewer_node)
    builder.add_node("human_review", human_review)
    builder.add_node("advance_pr", advance_pr)

    # Edges
    builder.set_entry_point("architect")

    # Architect → Fan Out (dynamic edge using Send)
    builder.add_conditional_edges("architect", fan_out_to_developers)

    # Fan In: all developer Send nodes converge to reviewer
    builder.add_edge("developer", "reviewer")

    # Reviewer → conditionally retry or go to human gate
    builder.add_conditional_edges(
        "reviewer",
        should_continue_after_review,
        {"fan_out": "architect", "human_review": "human_review"},
    )

    # Human gate → advance or retry
    builder.add_conditional_edges(
        "human_review",
        should_continue_after_human,
        {"fan_out": "architect", "advance_pr": "advance_pr", END: END},
    )

    # After advancing PR, fan out again for next task
    builder.add_conditional_edges("advance_pr", fan_out_to_developers)

    return builder.compile()
