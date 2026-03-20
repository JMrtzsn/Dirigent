"""Reviewer node — evaluates all developer results and selects the best.

In the real implementation, this will:
1. Check out each developer's branch
2. Run the test suite on each
3. Evaluate architectural alignment against the plan
4. Compare diff sizes
5. Produce a recommendation
"""

from __future__ import annotations

import logging

from dirigent.state import (
    DeveloperStatus,
    GraphState,
    ReviewResult,
    ReviewVerdict,
)

logger = logging.getLogger(__name__)


def reviewer_node(state: GraphState) -> dict:
    """Review all developer results and select the best implementation.

    Returns:
        State update with review containing verdicts and a recommendation.
    """
    results = state.developer_results
    logger.info("Reviewer evaluating %d developer results", len(results))

    # Filter to successful results only
    successful = [r for r in results if r.status == DeveloperStatus.SUCCESS]

    if not successful:
        logger.warning("No successful developer results — requesting retry")
        return {
            "review": ReviewResult(
                verdicts=[],
                selected_developer_id="",
                recommendation="All developers failed. Retry recommended.",
                should_retry=True,
            )
        }

    # STUB: Replace with LLM-powered review that actually checks branches
    verdicts = _stub_review(successful)
    best = max(verdicts, key=lambda v: v.architectural_alignment + v.diff_size_score)

    review = ReviewResult(
        verdicts=verdicts,
        selected_developer_id=best.developer_id,
        recommendation=f"Selected {best.developer_id}: best architecture and diff balance.",
        should_retry=False,
    )

    _log_comparison_table(verdicts, best.developer_id)

    return {"review": review}


def _stub_review(results: list) -> list[ReviewVerdict]:
    """Placeholder review logic. Replace with real branch evaluation."""
    verdicts = []
    for i, result in enumerate(results):
        verdicts.append(
            ReviewVerdict(
                developer_id=result.developer_id,
                passed_tests=True,
                architectural_alignment=4 - i,  # Stub: first dev scores highest
                diff_size_score=3 + i,  # Stub: last dev has smallest diff
                notes=f"Stub review for {result.developer_id}",
            )
        )
    return verdicts


def _log_comparison_table(verdicts: list[ReviewVerdict], selected_id: str) -> None:
    """Log a comparison table of all developer results."""
    header = f"{'Developer':<12} {'Tests':>6} {'Arch':>5} {'Diff':>5} {'Selected':>9}"
    logger.info(header)
    logger.info("-" * len(header))
    for v in verdicts:
        marker = "  <<<" if v.developer_id == selected_id else ""
        logger.info(
            f"{v.developer_id:<12} {'PASS' if v.passed_tests else 'FAIL':>6} "
            f"{v.architectural_alignment:>5} {v.diff_size_score:>5}{marker}"
        )
