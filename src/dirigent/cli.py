"""CLI entry point for Dirigent."""

from __future__ import annotations

import argparse
import logging
import sys

from dirigent.graph import build_graph
from dirigent.llm.config import Config
from dirigent.state import GraphState


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

    graph = build_graph()

    initial_state = GraphState(
        objective=args.objective,
        repo_path=args.repo,
    )

    logging.info("Starting Dirigent with objective: %s", args.objective)
    logging.info("Target repo: %s | Developers: %d", args.repo, args.developers)

    # Build config — nodes pull this from RunnableConfig["configurable"]["dirigent_config"]
    dirigent_config = None
    if not args.no_llm:
        try:
            dirigent_config = Config()
            logging.info("LLM provider initialized: %s", type(dirigent_config.provider).__name__)
        except Exception:
            logging.exception("Failed to initialize LLM provider, running with stubs")

    config = {
        "configurable": {
            "thread_id": "dirigent-main",
            "dirigent_config": dirigent_config,
        }
    }

    try:
        for event in graph.stream(initial_state, config=config, stream_mode="updates"):
            for node_name, _update in event.items():
                logging.info("Node '%s' completed", node_name)
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
        sys.exit(1)

    logging.info("Dirigent run complete")


if __name__ == "__main__":
    main()
