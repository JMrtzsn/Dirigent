"""CLI entry point for Dirigent."""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
from pathlib import Path

from dirigent.orchestrator import Orchestrator, OrchestratorConfig
from dirigent.utils.worktree import WorktreeManager

logger = logging.getLogger(__name__)


def _register_signal_handlers(repo_path: str) -> None:
    """Register SIGINT/SIGTERM handlers that clean up worktrees on exit."""
    manager = WorktreeManager(Path(repo_path))

    def _handle_signal(signum: int, _frame: object) -> None:
        sig_name = signal.Signals(signum).name
        print(f"\n[dirigent] Received {sig_name}, cleaning up worktrees...", file=sys.stderr)
        manager.cleanup_all()
        sys.exit(128 + signum)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)


async def _human_review(summary: str) -> bool:
    """Prompt the user for approval (blocking input in a thread)."""
    loop = asyncio.get_running_loop()

    def _ask() -> bool:
        print(f"\n{'=' * 60}", file=sys.stderr)
        print("HUMAN REVIEW", file=sys.stderr)
        print(f"{'=' * 60}", file=sys.stderr)
        print(summary, file=sys.stderr)
        response = input("\nApprove? [y/N]: ").strip().lower()
        return response in ("y", "yes")

    return await loop.run_in_executor(None, _ask)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="dirigent",
        description="Agent-to-agent interaction framework — SLIB orchestrator",
    )
    parser.add_argument(
        "objective",
        help="The high-level objective to decompose and implement",
    )
    parser.add_argument(
        "--repo",
        default=".",
        help="Path to the target git repository (default: current directory)",
    )
    parser.add_argument(
        "--developers",
        type=int,
        default=3,
        help="Number of parallel developer agents (default: 3)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    repo_path = str(Path(args.repo).resolve())
    _register_signal_handlers(repo_path)

    config = OrchestratorConfig(
        repo_path=repo_path,
        objective=args.objective,
        num_developers=args.developers,
        human_approve_callback=_human_review,
    )

    orchestrator = Orchestrator(config)

    print(f"[dirigent] Objective: {args.objective}", file=sys.stderr)
    print(f"[dirigent] Repo: {repo_path}", file=sys.stderr)
    print(f"[dirigent] Developers: {args.developers}", file=sys.stderr)
    print(file=sys.stderr)

    try:
        pr_url = asyncio.run(orchestrator.run())
    except KeyboardInterrupt:
        print("\n[dirigent] Interrupted", file=sys.stderr)
        sys.exit(1)

    if pr_url:
        print(f"\n[dirigent] Draft PR: {pr_url}", file=sys.stderr)
    else:
        print("\n[dirigent] Pipeline failed — see logs above", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
