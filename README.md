# Dirigent

> *Dirigent (noun): a conductor — one who directs an orchestra.*

A fan-out/fan-in AI agent orchestrator built with [LangGraph](https://langchain-ai.github.io/langgraph/). Dirigent decomposes complex software tasks into small, reviewable PRs, spawns multiple AI developers in parallel to implement each one, then selects the best solution through automated review.

## Why

Most AI coding tools run a single agent sequentially. Dirigent takes a different approach:

1. **Architect** breaks a task into atomic PRs (small changes, easy to review)
2. **Fan Out** — N developer agents work the same sub-task in parallel, each in an isolated git worktree
3. **Fan In** — a Reviewer agent tests, compares, and selects the best implementation
4. **Human Stop** — you verify and approve before anything gets committed
5. **Repeat** for the next PR in the plan

This is "best of N" code generation with git isolation and human oversight.

## Architecture

```
                          ┌─────────────────────────────────────────────┐
                          │              DIRIGENT GRAPH                  │
                          │                                             │
                          │  ┌───────────┐                              │
                   ───────┼─▶│ Architect  │  Splits objective into      │
                          │  │            │  2-5 PR-sized sub-tasks     │
                          │  └─────┬──────┘                              │
                          │        │                                     │
                          │        │ Send() × N                          │
                          │        ▼                                     │
                          │  ┌───────────┐ ┌───────────┐ ┌───────────┐  │
                          │  │Developer 0│ │Developer 1│ │Developer 2│  │
                          │  │ worktree  │ │ worktree  │ │ worktree  │  │
                          │  │ branch/0  │ │ branch/1  │ │ branch/2  │  │
                          │  └─────┬──────┘ └─────┬──────┘ └─────┬──────┘  │
                          │        │              │              │       │
                          │        └──────────────┼──────────────┘       │
                          │                       │ Fan In               │
                          │                       ▼                      │
                          │               ┌───────────┐                  │
                          │               │ Reviewer   │  Tests, scores, │
                          │               │            │  selects best   │
                          │               └─────┬──────┘                  │
                          │                     │                        │
                          │          ┌──────────┼──────────┐             │
                          │          ▼          │          ▼             │
                          │    [retry: reject]  │   [human_review]      │
                          │     back to arch.   │    interrupt()        │
                          │                     │          │             │
                          │                     │    ┌─────┼──────┐      │
                          │                     │    ▼     │     ▼      │
                          │                     │ [reject] │ [approve]  │
                          │                     │  retry   │  next PR   │
                          │                     │          │  or END    │
                          └─────────────────────┼──────────┼────────────┘
                                                │          │
                                                ▼          ▼
                                           loops back    done
```

### Key Concepts

| Concept | Implementation | Why |
|---|---|---|
| **Fan Out** | LangGraph `Send` API | Spawns N developer nodes in parallel, each receiving its own input |
| **Fan In** | Custom reducer on `developer_results` | Merges parallel results by `developer_id` into a single list |
| **Git Isolation** | `git worktree` per developer | Parallel agents modify files without conflicting |
| **Human Gate** | LangGraph `interrupt()` | Pauses execution, resumes only after manual approval |
| **Conditional Routing** | Graph edges with predicates | Reviewer can reject (retry fan-out) or proceed; human can reject or approve |

## Project Structure

```
Dirigent/
├── Makefile                    # Dev commands: make dev, make test, make lint, make check
├── pyproject.toml              # Build config, dependencies, tooling
├── README.md
├── .gitignore
│
├── src/dirigent/
│   ├── __init__.py
│   ├── state.py                # GraphState, SubTask, DeveloperResult, ReviewResult, reducers
│   ├── graph.py                # LangGraph StateGraph definition, topology, optional checkpointer
│   ├── cli.py                  # CLI entry point: SqliteSaver, signal handlers, --db/--no-llm flags
│   │
│   ├── nodes/
│   │   ├── architect.py        # LLM-powered repo analysis → PR plan (JSON parse + stub fallback)
│   │   ├── developer.py        # LLM code gen in worktrees: file ops, commit, test, cleanup
│   │   └── reviewer.py         # LLM verdict scoring with heuristic fallback
│   │
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── provider.py         # LLMProvider protocol, Message, CompletionResult, ProviderError
│   │   ├── copilot.py          # CopilotProvider: OpenAI SDK + Copilot auth + required headers
│   │   └── config.py           # Config, Role, ModelConfig, default model assignments
│   │
│   └── utils/
│       ├── __init__.py
│       ├── worktree.py         # Git worktree create/remove/cleanup lifecycle
│       └── repo.py             # File tree scanning, key file reading, repo context builder
│
└── tests/
    ├── test_state.py           # State construction and reducer logic (6 tests)
    ├── test_llm.py             # Provider protocol, token resolution, CopilotProvider (17 tests)
    ├── test_nodes.py           # All node tests: architect, developer, reviewer (38 tests)
    ├── test_repo.py            # File tree scanning, key files, context building (6 tests)
    └── test_integration.py     # Full graph end-to-end: stub, LLM, human approval (3 tests)
```

## State Flow

The `GraphState` dataclass flows through every node. Key fields:

```python
@dataclass
class GraphState:
    objective: str                  # What the user wants done
    repo_path: str                  # Target git repository
    plan: list[SubTask]             # Architect's decomposition
    current_pr_index: int           # Which PR we're working on
    developer_results: list[...]    # Fan-in merged results (Annotated reducer)
    review: ReviewResult            # Reviewer's verdict and selection
    human_approved: bool            # Human gate signal
    iteration: int                  # Loop counter
```

The `developer_results` field uses an `Annotated` custom reducer (`merge_developer_results`) that merges results from parallel developers by `developer_id`, so the Reviewer always sees the latest result from each developer.

## Current Status

All nodes are fully wired with LLM-powered implementations. The system uses GitHub Copilot as its LLM provider (OpenAI-compatible API, $0 under Copilot subscription). Each node falls back to stub behavior on LLM errors, so the graph always completes.

| Component | Status | Description |
|---|---|---|
| Graph topology | **Done** | Full `Architect → Fan Out → Fan In → Reviewer → Human → Loop` |
| State schema | **Done** | Typed dataclasses with LangGraph-compatible reducers |
| Architect node | **Done** | LLM-powered repo analysis and PR decomposition with JSON parsing |
| Developer node | **Done** | LLM code generation in isolated git worktrees, file operations, auto-commit, test execution |
| Reviewer node | **Done** | LLM-powered verdict scoring with heuristic fallback |
| Human gate | **Done** | Uses `interrupt()` — pauses for real input, resumes with `Command(resume=)` |
| Git worktree util | **Done** | Full create/remove/cleanup lifecycle |
| Checkpointing | **Done** | SQLite-backed persistence — runs survive crashes |
| Signal handlers | **Done** | SIGINT/SIGTERM clean up worktrees before exit |
| LLM provider | **Done** | GitHub Copilot API with auto-token resolution from OpenCode auth cache |
| Repo context | **Done** | File tree scanning + key file reading for architect context |
| CLI | **Done** | `dirigent "objective" --repo /path --developers 3 --db path --no-llm` |
| Tests | **Done** | 72 tests: unit, integration, full graph end-to-end |

## Quickstart

```bash
# Clone and set up
git clone <repo-url> && cd Dirigent
make dev        # Creates venv + installs all deps

# Verify
make check      # Runs lint + tests (72 tests)

# Run with stubs (no LLM calls, no API token needed)
dirigent "Refactor the auth module" --repo /path/to/repo --no-llm

# Run with LLM (requires GitHub Copilot subscription)
# Token is auto-resolved from OpenCode's auth cache (~/.local/share/opencode/auth.json)
# or from DIRIGENT_COPILOT_TOKEN env var
dirigent "Refactor the auth module" --repo /path/to/repo --developers 3 -v
```

### CLI Options

```
dirigent <objective> [options]

positional:
  objective              The high-level task to decompose and implement

options:
  --repo PATH            Target git repository (default: .)
  --developers N         Number of parallel developer agents (default: 3)
  --no-llm               Run with stub nodes (no LLM calls)
  --db PATH              SQLite checkpoint database (default: .dirigent/checkpoints.db)
  -v, --verbose          Enable debug logging
```

## Make Targets

```
  help            Show this help
  venv            Create virtual environment
  install         Install production dependencies
  dev             Install all dependencies (dev included)
  lint            Run ruff linter
  lint-fix        Run ruff with auto-fix
  format          Format code with ruff
  test            Run test suite
  check           Run lint + tests (CI gate)
  run             Run Dirigent with a sample objective
  clean           Remove build artifacts and venv
```

## Roadmap

### Done

- [x] LLM provider abstraction (GitHub Copilot API, OpenAI-compatible)
- [x] Architect node: LLM-powered repo analysis and PR decomposition
- [x] Developer node: LLM code generation in isolated git worktrees
- [x] Reviewer node: LLM-powered verdict scoring with heuristic fallback
- [x] SQLite checkpointer for crash recovery
- [x] SIGINT/SIGTERM signal handlers for worktree cleanup
- [x] Full integration tests (stub, LLM, human approval flow)

### Next

- [ ] End-to-end test with real Copilot API calls against a test repo
- [ ] Retry logic with exponential backoff for LLM calls
- [ ] Timeout handling for developer nodes
- [ ] Streaming output during LLM calls

### Future

- [ ] LangSmith tracing integration
- [ ] Structured logging with run IDs
- [ ] Terminal UI with progress indicators for parallel developers
- [ ] Cost tracking per run
- [ ] Developer specialization (different prompts/models per developer)
- [ ] Reviewer voting (multiple reviewers with consensus)
- [ ] PR auto-creation via `gh` CLI

## Design Decisions

**Why LangGraph over raw asyncio?**
LangGraph gives us `Send` (typed fan-out), `interrupt` (human-in-the-loop), `Annotated` reducers (fan-in merge), and checkpointing for free. Building this from scratch would mean reimplementing half of what LangGraph already provides.

**Why git worktrees over branches?**
Branches share a working directory. If two agents write to the same file simultaneously, you get corruption. Worktrees give each agent its own filesystem-level copy of the repo. This is the correct isolation primitive for parallel code generation.

**Why "best of N" over single-agent?**
For well-defined tasks, a single agent is fine. For ambiguous tasks (refactoring, architecture decisions), solution quality varies significantly between runs. Running N agents and selecting the best is a simple way to push the quality frontier without more sophisticated prompting.

**Why GitHub Copilot API?**
Free under a Copilot subscription. OpenAI-compatible, so the provider is trivially swappable. Uses `claude-sonnet-4.6` for architect/reviewer (strong reasoning) and `claude-haiku-4.5` for developers (cheap for parallel fan-out).

**Why stubs as fallback?**
Every node falls back to deterministic stub behavior on LLM errors. This means the graph always completes — useful for testing, CI, and demos without API credentials. The `--no-llm` flag forces stub mode explicitly.

## License

MIT
