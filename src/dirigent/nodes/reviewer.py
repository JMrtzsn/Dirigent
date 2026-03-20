"""Reviewer node — evaluates all developer results and selects the best.

Calls the LLM with each developer's diff stats and test output to produce
scored verdicts and a selection recommendation.
Falls back to heuristic scoring if no LLM config is available.
"""

# NOTE: Do NOT use `from __future__ import annotations` in node files.
# LangGraph introspects the `config: RunnableConfig` parameter type at runtime.
# See architect.py for details.

import json
import logging
import re

from langchain_core.runnables import RunnableConfig

from dirigent.llm.config import Config, Role
from dirigent.llm.provider import Message, ProviderError
from dirigent.state import (
    DeveloperResult,
    DeveloperStatus,
    GraphState,
    ReviewResult,
    ReviewVerdict,
)

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a Code Reviewer. You evaluate multiple developer implementations of \
the same sub-task and select the best one.

You will receive a summary of each developer's work including:
- Developer ID
- Diff statistics (files changed, lines added/removed)
- Test output (pass/fail, details)
- Any errors encountered

Score each developer on two axes (1-5 scale):
- architectural_alignment: How well does the implementation follow good \
software design? (SOLID, clean code, minimal coupling)
- diff_size_score: How focused and minimal is the change? (5 = small and \
focused, 1 = large and sprawling)

Respond with ONLY a JSON object. No markdown, no explanation. Example:

{
  "verdicts": [
    {
      "developer_id": "dev-0",
      "passed_tests": true,
      "architectural_alignment": 4,
      "diff_size_score": 5,
      "notes": "Clean implementation, minimal changes"
    },
    {
      "developer_id": "dev-1",
      "passed_tests": true,
      "architectural_alignment": 3,
      "diff_size_score": 3,
      "notes": "Works but touches too many files"
    }
  ],
  "selected_developer_id": "dev-0",
  "recommendation": "dev-0 has the cleanest implementation with focused changes"
}

Rules:
- Always select exactly one developer as the winner
- If all developers failed tests, set selected_developer_id to "" and \
recommend retry
- Prefer implementations that pass tests and have smaller, focused diffs
- Be specific in notes about what was good or bad
"""


def _get_config(runnable_config: RunnableConfig | None) -> Config | None:
    """Extract Dirigent Config from LangGraph's RunnableConfig."""
    if runnable_config is None:
        return None
    configurable = runnable_config.get("configurable", {})
    return configurable.get("dirigent_config")


def reviewer_node(state: GraphState, config: RunnableConfig | None = None) -> dict:
    """Review all developer results and select the best implementation.

    If a Dirigent Config with an LLM provider is available (via RunnableConfig),
    calls the LLM. Otherwise falls back to heuristic scoring.

    Args:
        state: Current graph state with developer_results populated.
        config: LangGraph runnable config containing dirigent_config.

    Returns:
        State update with review containing verdicts and a recommendation.
    """
    results = state.developer_results
    logger.info("Reviewer evaluating %d developer results", len(results))

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

    dirigent_config = _get_config(config)
    if dirigent_config is None:
        logger.info("No LLM config available, using heuristic review")
        verdicts = _stub_review(successful)
    else:
        verdicts = _llm_review(results, dirigent_config)

    if not verdicts:
        verdicts = _stub_review(successful)

    best = max(
        verdicts,
        key=lambda v: v.architectural_alignment + v.diff_size_score,
    )

    review = ReviewResult(
        verdicts=verdicts,
        selected_developer_id=best.developer_id,
        recommendation=f"Selected {best.developer_id}: best architecture and diff balance.",
        should_retry=False,
    )

    _log_comparison_table(verdicts, best.developer_id)
    return {"review": review}


def _llm_review(results: list[DeveloperResult], config: Config) -> list[ReviewVerdict]:
    """Use the LLM to score developer implementations."""
    summaries: list[str] = []
    for r in results:
        status = "SUCCESS" if r.status == DeveloperStatus.SUCCESS else "FAILED"
        summary = (
            f"### {r.developer_id} ({status})\n"
            f"**Branch:** {r.branch_name}\n"
            f"**Diff stats:** {r.diff_stats or '(none)'}\n"
            f"**Test output:** {r.test_output or '(none)'}\n"
            f"**Errors:** {r.error or '(none)'}\n"
        )
        summaries.append(summary)

    user_prompt = "## Developer Results\n\n" + "\n".join(summaries)

    model_config = config.model_for(Role.REVIEWER)
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
        logger.exception("LLM review call failed, falling back to heuristic")
        return []

    return _parse_review(result.content)


def _parse_review(raw_response: str) -> list[ReviewVerdict]:
    """Parse the LLM's JSON response into ReviewVerdict objects.

    Returns empty list on parse failure (caller falls back to stub).
    """
    cleaned = raw_response.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("Failed to parse reviewer LLM response as JSON")
        return []

    if not isinstance(data, dict) or "verdicts" not in data:
        logger.warning("Reviewer LLM response missing 'verdicts' key")
        return []

    verdicts: list[ReviewVerdict] = []
    for item in data["verdicts"]:
        if not isinstance(item, dict):
            continue
        try:
            verdicts.append(
                ReviewVerdict(
                    developer_id=str(item["developer_id"]),
                    passed_tests=bool(item.get("passed_tests", False)),
                    architectural_alignment=int(item.get("architectural_alignment", 3)),
                    diff_size_score=int(item.get("diff_size_score", 3)),
                    notes=str(item.get("notes", "")),
                )
            )
        except (KeyError, ValueError) as exc:
            logger.warning("Skipping malformed verdict: %s (%s)", item, exc)

    return verdicts


def _stub_review(results: list[DeveloperResult]) -> list[ReviewVerdict]:
    """Heuristic scoring for testing without an LLM."""
    verdicts = []
    for i, result in enumerate(results):
        verdicts.append(
            ReviewVerdict(
                developer_id=result.developer_id,
                passed_tests=True,
                architectural_alignment=4 - i,  # First dev scores highest
                diff_size_score=3 + i,  # Last dev has smallest diff
                notes=f"Heuristic review for {result.developer_id}",
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
            "%s %s %d %d%s",
            f"{v.developer_id:<12}",
            f"{'PASS' if v.passed_tests else 'FAIL':>6}",
            v.architectural_alignment,
            v.diff_size_score,
            marker,
        )
