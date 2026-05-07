"""Review loop pattern — submit work, get feedback, revise, repeat.

Implements a structured critique/revision cycle between a producer agent
and a reviewer agent, with a configurable max number of rounds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from dirigent.runtime import Runtime


@dataclass
class ReviewResult:
    """Outcome of a review loop."""

    accepted: bool
    final_result: Any
    rounds: int
    history: list[dict[str, Any]] = field(default_factory=list)


async def review_loop(
    runtime: Runtime,
    *,
    producer: str = "",
    producer_capability: str = "",
    reviewer: str = "",
    reviewer_capability: str = "",
    task: str,
    context: dict[str, Any] | None = None,
    max_rounds: int = 3,
    sender: str = "__review_loop__",
) -> ReviewResult:
    """Run a produce → review → revise cycle.

    The producer generates work, the reviewer critiques it. If rejected,
    the reviewer's feedback is passed back to the producer as context
    for revision. Repeats up to max_rounds.
    """
    history: list[dict[str, Any]] = []
    current_context = dict(context or {})

    for round_num in range(1, max_rounds + 1):
        # Producer generates
        production = await runtime.delegate(
            to=producer,
            capability=producer_capability,
            task=task,
            context=current_context,
            sender=sender,
        )

        history.append({
            "round": round_num,
            "phase": "produce",
            "success": production.success,
            "result": production.result,
        })

        if not production.success:
            return ReviewResult(
                accepted=False,
                final_result=production.result,
                rounds=round_num,
                history=history,
            )

        # Reviewer evaluates
        review_context = {
            "submission": production.result,
            "original_task": task,
            "round": round_num,
            **current_context,
        }

        review = await runtime.delegate(
            to=reviewer,
            capability=reviewer_capability,
            task=f"Review this submission for: {task}",
            context=review_context,
            sender=sender,
        )

        history.append({
            "round": round_num,
            "phase": "review",
            "success": review.success,
            "result": review.result,
        })

        # Check if accepted (reviewer result should be a dict with 'accepted' key)
        accepted = False
        feedback = {}
        if isinstance(review.result, dict):
            accepted = review.result.get("accepted", False)
            feedback = review.result.get("feedback", {})
        elif review.success:
            # If reviewer returns a simple truthy value, treat as accepted
            accepted = bool(review.result)

        if accepted:
            return ReviewResult(
                accepted=True,
                final_result=production.result,
                rounds=round_num,
                history=history,
            )

        # Feed back for next round
        current_context["previous_attempt"] = production.result
        current_context["feedback"] = feedback

    # Exhausted rounds
    return ReviewResult(
        accepted=False,
        final_result=production.result,
        rounds=max_rounds,
        history=history,
    )
