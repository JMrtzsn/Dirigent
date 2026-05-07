"""Example: Three agents collaborate to design and review a CLI tool.

Demonstrates the core Dirigent patterns:
- Capability-based delegation
- Fan-out to multiple agents
- Review loop (produce → critique → revise)

Run:
    export DIRIGENT_COPILOT_TOKEN="your-token"
    python examples/cli_tool_design.py
"""

import asyncio

from dirigent import Runtime
from dirigent.llm import CopilotLLMProvider, LLMAgent
from dirigent.patterns import fanout, review_loop


async def main() -> None:
    provider = CopilotLLMProvider()

    # Define specialized agents
    architect = LLMAgent(
        name="architect",
        capabilities=["design"],
        provider=provider,
        model="claude-sonnet-4.6",
        system_prompt=(
            "You are a software architect. When asked to design something, "
            "respond with a clear, concise technical design as a JSON object with keys: "
            "modules (list of module names), interfaces (list of interface descriptions), "
            "data_flow (description of how data moves through the system)."
        ),
    )

    developer_a = LLMAgent(
        name="dev-a",
        capabilities=["implement"],
        provider=provider,
        model="claude-haiku-4.5",
        system_prompt=(
            "You are a pragmatic developer who favors simplicity. "
            "Implement the requested feature with minimal abstractions. "
            "Respond with a JSON object: {files: [{path, content}]}"
        ),
    )

    developer_b = LLMAgent(
        name="dev-b",
        capabilities=["implement"],
        provider=provider,
        model="claude-haiku-4.5",
        system_prompt=(
            "You are a developer who favors robustness and error handling. "
            "Implement the requested feature with thorough validation. "
            "Respond with a JSON object: {files: [{path, content}]}"
        ),
    )

    reviewer = LLMAgent(
        name="reviewer",
        capabilities=["review"],
        provider=provider,
        model="claude-sonnet-4.6",
        system_prompt=(
            "You are a code reviewer. Evaluate the submission against the original task. "
            "Respond with a JSON object: "
            '{accepted: true/false, feedback: {strengths: [...], issues: [...]}}'
        ),
    )

    # Wire up the runtime
    rt = Runtime()
    rt.register(architect, developer_a, developer_b, reviewer)

    # 1. Architect designs the CLI tool
    print("=== Phase 1: Design ===")
    design = await rt.delegate(
        to="architect",
        task="Design a CLI tool that converts CSV files to JSON with filtering support",
    )
    print(f"Design: {design.result}\n")

    # 2. Fan out to both developers
    print("=== Phase 2: Implement (fan-out) ===")
    implementations = await fanout(
        rt,
        agents=["dev-a", "dev-b"],
        task="Implement a CLI tool that converts CSV to JSON with column filtering",
        context={"design": design.result},
    )
    for impl in implementations:
        print(f"{impl.sender}: success={impl.success}")
    print()

    # 3. Review loop with dev-a
    print("=== Phase 3: Review loop ===")
    result = await review_loop(
        rt,
        producer="dev-a",
        reviewer="reviewer",
        task="Implement a CSV-to-JSON CLI tool with --columns flag for filtering",
        max_rounds=2,
    )
    print(f"Accepted: {result.accepted}")
    print(f"Rounds: {result.rounds}")
    print(f"Final result preview: {str(result.final_result)[:200]}...")

    # 4. Show traces
    print(f"\n=== Traces: {len(rt.traces)} interactions recorded ===")
    for trace in rt.traces:
        print(f"  {trace['sender']} -> {trace['receiver']}: success={trace['success']}")


if __name__ == "__main__":
    asyncio.run(main())
