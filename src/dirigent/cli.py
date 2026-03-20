"""CLI entry point for Dirigent."""

from __future__ import annotations

import argparse
import logging
import signal
import sys
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from dirigent.graph import build_graph
from dirigent.llm.config import Config
from dirigent.state import GraphState
from dirigent.utils.worktree import WorktreeManager

logger = logging.getLogger(__name__)

_DEFAULT_DB = ".dirigent/checkpoints.db"


def _register_signal_handlers(repo_path: str) -> None:
    """Register SIGINT/SIGTERM handlers that clean up worktrees on exit."""
    manager = WorktreeManager(Path(repo_path))

    def _handle_signal(signum: int, _frame: object) -> None:
        sig_name = signal.Signals(signum).name
        logger.info("Received %s, cleaning up worktrees...", sig_name)
        manager.cleanup_all()
        sys.exit(128 + signum)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)


def main() -> None:
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
        "--no-llm",
        action="store_true",
        help="Run with stub nodes (no LLM calls)",
    )
    parser.add_argument(
        "--db",
        default=_DEFAULT_DB,
        help=f"SQLite checkpoint database path (default: {_DEFAULT_DB})",
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

    # Ensure checkpoint directory exists
    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Register cleanup handlers before any worktrees are created
    _register_signal_handlers(args.repo)

    # Build graph with SQLite checkpointer for persistent state
    with SqliteSaver.from_conn_string(str(db_path)) as checkpointer:
        graph = build_graph(checkpointer=checkpointer)

        initial_state = GraphState(
            objective=args.objective,
            repo_path=args.repo,
        )

        logger.info("Starting Dirigent with objective: %s", args.objective)
        logger.info("Target repo: %s | Developers: %d", args.repo, args.developers)
        logger.info("Checkpoint DB: %s", db_path)

        # Build config — nodes pull this from RunnableConfig["configurable"]
        dirigent_config = None
        if not args.no_llm:
            try:
                dirigent_config = Config()
                logger.info(
                    "LLM provider initialized: %s",
                    type(dirigent_config.provider).__name__,
                )
            except Exception:
                logger.exception("Failed to initialize LLM provider, running with stubs")

        config = {
            "configurable": {
                "thread_id": "dirigent-main",
                "dirigent_config": dirigent_config,
            }
        }

        try:
            for event in graph.stream(initial_state, config=config, stream_mode="updates"):
                for node_name, _update in event.items():
                    logger.info("Node '%s' completed", node_name)
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
            sys.exit(1)

    logger.info("Dirigent run complete")


if __name__ == "__main__":
    main()
