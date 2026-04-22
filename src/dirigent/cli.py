"""CLI entry point for Dirigent."""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import warnings
from pathlib import Path

from langgraph.checkpoint.memory import MemorySaver

from dirigent.graph import build_graph
from dirigent.llm.config import Config
from dirigent.state import GraphState
from dirigent.utils import display
from dirigent.utils.worktree import WorktreeManager

logger = logging.getLogger(__name__)

# Nodes that produce phase transitions worth labeling
_PHASE_LABELS = {
    "architect": "Planning",
    "setup_feature_branch": "Setup",
    "reviewer": "Review",
    "human_review": "Human Review",
    "merge_winner": "Merge",
    "finalize": "Finalize",
}


def _register_signal_handlers(repo_path: str) -> None:
    """Register SIGINT/SIGTERM handlers that clean up worktrees on exit."""
    manager = WorktreeManager(Path(repo_path))

    def _handle_signal(signum: int, _frame: object) -> None:
        sig_name = signal.Signals(signum).name
        display.error(f"Received {sig_name}, cleaning up worktrees...")
        manager.cleanup_all()
        sys.exit(128 + signum)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)


def main() -> None:
    warnings.filterwarnings("ignore", message=".*Pydantic V1.*")

    parser = argparse.ArgumentParser(
        prog="dirigent",
        description="Fan-out/fan-in AI agent orchestrator",
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
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    _register_signal_handlers(args.repo)

    checkpointer = MemorySaver()
    graph = build_graph(checkpointer=checkpointer)

    initial_state = GraphState(
        objective=args.objective,
        repo_path=args.repo,
    )

    display.banner(args.objective, args.repo, args.developers)

    try:
        dirigent_config = Config()
    except Exception:
        display.error("Failed to initialize LLM provider")
        logger.debug("LLM init failure", exc_info=True)
        sys.exit(1)

    config = {
        "configurable": {
            "thread_id": "dirigent-main",
            "dirigent_config": dirigent_config,
        }
    }

    last_phase: str | None = None
    try:
        for event in graph.stream(initial_state, config=config, stream_mode="updates"):
            for node_name, update in event.items():
                # Print phase separator on phase transitions
                phase = _PHASE_LABELS.get(node_name)
                if phase and phase != last_phase and node_name != "reviewer":
                    display.phase_separator(phase)
                    last_phase = phase

                # Print the review table before the reviewer checkmark
                if node_name == "reviewer":
                    display.phase_separator("Review")
                    last_phase = "Review"
                    display.review_table(update)

                display.node_event(node_name, update)
    except KeyboardInterrupt:
        display.error("Interrupted by user")
        sys.exit(1)

    display.done()


if __name__ == "__main__":
    main()
