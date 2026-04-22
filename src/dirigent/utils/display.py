"""Terminal display utilities for Dirigent.

Provides ANSI-colored, structured output for the orchestration pipeline.
All output goes through stderr so it doesn't interfere with piped stdout.
"""

from __future__ import annotations

import sys
from typing import Any, TextIO

# ANSI color codes
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_CYAN = "\033[36m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_MAGENTA = "\033[35m"
_BLUE = "\033[34m"

# Node label colors (resolved at render time via _icon())
_ICON_COLORS = {
    "architect": _CYAN,
    "setup_feature_branch": _BLUE,
    "developer": _GREEN,
    "reviewer": _MAGENTA,
    "human_review": _YELLOW,
    "merge_winner": _GREEN,
    "advance_pr": _BLUE,
    "finalize": _CYAN,
    "__interrupt__": _YELLOW,
}

_ICON_LABELS = {
    "setup_feature_branch": "branch",
    "human_review": "human",
    "merge_winner": "merge",
    "advance_pr": "next",
    "__interrupt__": "waiting",
}


def _icon(node_name: str) -> str:
    """Render a colored node label, respecting color support."""
    label = _ICON_LABELS.get(node_name, node_name)
    color = _ICON_COLORS.get(node_name, "")
    return _c(color, f"[{label}]")


def _out() -> TextIO:
    return sys.stderr


def _supports_color() -> bool:
    return hasattr(_out(), "isatty") and _out().isatty()


def _c(code: str, text: str) -> str:
    if not _supports_color():
        return text
    return f"{code}{text}{_RESET}"


def _attr(obj: Any, name: str, default: Any = "") -> Any:
    """Get an attribute from a dataclass or dict."""
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, dict):
        return obj.get(name, default)
    return default


def banner(objective: str, repo: str, developers: int) -> None:
    """Print the startup banner."""
    w = _out()
    w.write("\n")
    line = "  ╔═══════════════════════════════════════╗"
    w.write(_c(_BOLD + _CYAN, line) + "\n")
    line = "  ║           D I R I G E N T             ║"
    w.write(_c(_BOLD + _CYAN, line) + "\n")
    line = "  ╚═══════════════════════════════════════╝"
    w.write(_c(_BOLD + _CYAN, line) + "\n")
    w.write("\n")
    w.write(f"  {_c(_DIM, 'Objective')}  {objective}\n")
    w.write(f"  {_c(_DIM, 'Repo')}       {repo}\n")
    w.write(f"  {_c(_DIM, 'Developers')} {developers}\n")
    w.write("\n")
    w.write(_c(_DIM, "  ─" * 20) + "\n\n")
    w.flush()


def node_event(node_name: str, update: dict) -> None:
    """Print a formatted event for a completed node."""
    icon = _icon(node_name)
    detail = _extract_detail(node_name, update)
    suffix = f"  {_c(_DIM, detail)}" if detail else ""
    _out().write(f"  {_c(_GREEN, '✓')} {icon}{suffix}\n")
    _out().flush()


def _extract_detail(node_name: str, update: dict) -> str:
    """Pull a meaningful one-liner from the node's state update."""
    if node_name == "architect":
        plan = update.get("plan", [])
        if plan:
            titles = [
                _attr(t, "title")
                for t in plan
            ]
            return f"{len(plan)} sub-tasks: {', '.join(titles)}"
    elif node_name == "setup_feature_branch":
        branch = update.get("feature_branch", "")
        if branch:
            return branch
    elif node_name == "developer":
        results = update.get("developer_results", [])
        if results:
            r = results[0] if isinstance(results, list) else results
            dev_id = _attr(r, "developer_id")
            status_val = _attr(r, "status")
            if hasattr(status_val, "value"):
                status_val = status_val.value
            return f"{dev_id} -> {status_val}"
    elif node_name == "reviewer":
        review = update.get("review")
        if review:
            selected = _attr(review, "selected_developer_id")
            if selected:
                return f"selected {selected}"
            return "requesting retry"
    elif node_name == "merge_winner":
        return "merged into feature branch"
    elif node_name == "finalize":
        return "draft PR created"
    elif node_name == "__interrupt__":
        return "awaiting human approval"
    return ""


def review_table(update: dict) -> None:
    """Print the reviewer's verdict table."""
    review = update.get("review")
    if not review:
        return

    verdicts = _attr(review, "verdicts", [])
    selected = _attr(review, "selected_developer_id", "")
    recommendation = _attr(review, "recommendation", "")

    w = _out()
    w.write("\n")
    hdr = (
        f"  {_c(_DIM, 'Developer')}"
        f"     {_c(_DIM, 'Tests')}"
        f"  {_c(_DIM, 'Arch')}"
        f"  {_c(_DIM, 'Diff')}"
    )
    w.write(hdr + "\n")
    w.write(f"  {_c(_DIM, '─' * 38)}\n")

    for v in verdicts:
        dev_id = _attr(v, "developer_id")
        passed = _attr(v, "passed_tests", False)
        arch = _attr(v, "architectural_alignment", 0)
        diff = _attr(v, "diff_size_score", 0)

        test_str = _c(_GREEN, "PASS") if passed else _c(_RED, "FAIL")
        marker = (
            _c(_GREEN + _BOLD, " <<<")
            if dev_id == selected
            else ""
        )
        w.write(
            f"  {dev_id:<14}{test_str}"
            f"   {arch}     {diff}{marker}\n"
        )

    if recommendation:
        w.write(
            f"\n  {_c(_DIM, 'Recommendation:')}"
            f" {recommendation}\n"
        )
    w.write("\n")
    w.flush()


def phase_separator(label: str) -> None:
    """Print a labeled separator between pipeline phases."""
    w = _out()
    pad = "─" * (34 - len(label))
    line = (
        f"\n  {_c(_DIM, '── ')}"
        f" {_c(_BOLD, label)}"
        f" {_c(_DIM, ' ' + pad)}\n\n"
    )
    w.write(line)
    w.flush()


def error(msg: str) -> None:
    """Print an error message."""
    _out().write(f"  {_c(_RED, '✗')} {msg}\n")
    _out().flush()


def success(msg: str) -> None:
    """Print a success message."""
    _out().write(f"  {_c(_GREEN, '✓')} {msg}\n")
    _out().flush()


def done() -> None:
    """Print the completion message."""
    w = _out()
    w.write(f"\n  {_c(_GREEN + _BOLD, '✓ Dirigent run complete')}\n\n")
    w.flush()
