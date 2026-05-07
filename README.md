# Dirigent

Agent-to-agent interaction framework for LLM-native systems.

Dirigent provides typed primitives for structured communication between AI agents: capability-based routing, delegation, fan-out, review loops, and observability. No chat-based free-for-all — agents delegate tasks, return structured results, and compose into reliable pipelines.

## Core Concepts

| Concept | What it does |
|---|---|
| **Agent** | Anything that handles a `DelegateRequest` and returns a `DelegateResponse` |
| **Capability** | Named unit of work an agent provides. Routing is capability-based. |
| **Runtime** | Execution engine — resolves capabilities, enforces timeouts, records traces |
| **Pattern** | Reusable interaction: delegation, fan-out, review loops |

## Quick Example

```python
import asyncio
from dirigent import Runtime, LLMAgent, CopilotLLMProvider
from dirigent.patterns import review_loop

provider = CopilotLLMProvider()

architect = LLMAgent(
    name="architect",
    capabilities=["design"],
    provider=provider,
    model="claude-sonnet-4.6",
    system_prompt="You are a software architect. Respond with JSON designs.",
)

developer = LLMAgent(
    name="developer",
    capabilities=["implement"],
    provider=provider,
    model="claude-haiku-4.5",
    system_prompt="You are a developer. Implement features as file operations.",
)

reviewer = LLMAgent(
    name="reviewer",
    capabilities=["review"],
    provider=provider,
    model="claude-sonnet-4.6",
    system_prompt="You are a reviewer. Respond with {accepted: bool, feedback: {...}}",
)

async def main():
    rt = Runtime()
    rt.register(architect, developer, reviewer)

    # Delegate by capability
    design = await rt.delegate(capability="design", task="Design a REST API for todos")

    # Review loop: produce → critique → revise (up to 3 rounds)
    result = await review_loop(
        rt,
        producer="developer",
        reviewer="reviewer",
        task="Implement the REST API",
        context={"design": design.result},
        max_rounds=3,
    )
    print(f"Accepted: {result.accepted}, Rounds: {result.rounds}")

asyncio.run(main())
```

## Patterns

### Delegate
Simple A→B task delegation with structured request/response.

### Fan-out
Send the same task to N agents in parallel, collect all results.

```python
from dirigent.patterns import fanout

results = await fanout(rt, agents=["dev-0", "dev-1", "dev-2"], task="implement auth")
```

### Review Loop
Produce → review → revise cycle with configurable max rounds.

```python
from dirigent.patterns import review_loop

result = await review_loop(rt, producer="dev", reviewer="reviewer", task="...", max_rounds=3)
```

## Architecture

```
dirigent/
├── core/           # Agent protocol, messages, capabilities, registry
├── llm/            # LLM-backed agents, provider protocol, Copilot implementation
├── patterns/       # Reusable interaction patterns (delegate, fanout, review)
├── observe/        # Tracing and per-agent metrics
├── runtime.py      # Execution engine
└── orchestrator.py # SLIB workflow (built on the framework itself)
```

Design principles:
- **Network-ready** — Messages are serializable dataclasses. Swap `InProcessChannel` for WebSocket/gRPC later.
- **Capability-based routing** — Callers request capabilities, not specific agents.
- **LLM-native** — `LLMAgent` handles prompt construction, LLM calls, response parsing. Subclass and override.
- **Single dependency** — Only requires `openai` SDK.

## Setup

```bash
git clone <repo-url> && cd Dirigent
make dev          # venv + deps
make check        # lint + tests
```

Token resolution (for `CopilotLLMProvider`):
1. `DIRIGENT_COPILOT_TOKEN` env var
2. OpenCode auth cache (`~/.local/share/opencode/auth.json`)

## Custom Agents

Implement the `Agent` protocol directly or use the `@agent` decorator:

```python
from dirigent import agent, DelegateRequest, DelegateResponse

@agent(capabilities=["summarize", "translate"])
class MyAgent:
    async def handle(self, request: DelegateRequest) -> DelegateResponse:
        # Your logic here
        return DelegateResponse(
            sender="myagent",
            receiver=request.sender,
            correlation_id=request.id,
            success=True,
            result={"summary": "..."},
        )
```

Or subclass `LLMAgent` for LLM-backed agents with custom prompting:

```python
from dirigent.llm import LLMAgent
from dirigent.core.message import DelegateRequest
from dirigent.llm.provider import LLMResponse

class CustomAgent(LLMAgent):
    def build_messages(self, request: DelegateRequest) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": "You are an expert at X."},
            {"role": "user", "content": request.task},
        ]

    def parse_response(self, llm_response: LLMResponse, request: DelegateRequest):
        return json.loads(llm_response.content)
```

## License

MIT
