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
├── Makefile                # Dev commands: make dev, make test, make lint
├── pyproject.toml          # Build config, dependencies, tooling
├── README.md
├── .gitignore
│
├── src/dirigent/
│   ├── __init__.py
│   ├── state.py            # GraphState schema, data classes, reducers
│   ├── graph.py            # LangGraph StateGraph definition and topology
│   ├── cli.py              # CLI entry point (`dirigent` command)
│   │
│   ├── nodes/
│   │   ├── architect.py    # Decomposes objective → list of SubTasks
│   │   ├── developer.py    # Executes a sub-task (receives input via Send)
│   │   └── reviewer.py     # Evaluates all developer results, picks best
│   │
│   └── utils/
│       └── worktree.py     # Git worktree create/remove/cleanup lifecycle
│
└── tests/
    ├── test_state.py       # State construction and reducer logic
    └── test_nodes.py       # Node input/output contracts
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

This is the **v0 skeleton**. All nodes return stubbed data. The graph topology, state management, reducers, and control flow are fully wired and tested.

| Component | Status | Description |
|---|---|---|
| Graph topology | **Done** | Full `Architect → Fan Out → Fan In → Reviewer → Human → Loop` |
| State schema | **Done** | Typed dataclasses with LangGraph-compatible reducers |
| Architect node | **Stub** | Returns hardcoded 3-PR plan |
| Developer node | **Stub** | Returns fake success result |
| Reviewer node | **Stub** | Scores developers with predictable numbers |
| Human gate | **Done** | Uses `interrupt()` — pauses for real input |
| Git worktree util | **Done** | Full create/remove/cleanup lifecycle |
| CLI | **Done** | `dirigent "objective" --repo /path --developers 3` |
| Tests | **Done** | 11 tests covering state, reducers, and node contracts |

## Quickstart

```bash
# Clone and set up
git clone <repo-url> && cd Dirigent
make dev        # Creates venv + installs all deps

# Verify
make check      # Runs lint + tests

# Run with stubbed data
make run        # Runs: dirigent "Refactor the auth module" --verbose
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

### Phase 1: Wire Real LLM Calls
- [ ] Add LLM provider abstraction (Copilot API / Anthropic / OpenAI)
- [ ] Replace `_stub_plan()` in architect with LLM-powered repo analysis and decomposition
- [ ] Replace `_stub_developer_work()` with actual code generation in git worktrees
- [ ] Replace `_stub_review()` with branch checkout, test execution, and LLM-powered comparison

### Phase 2: Robustness
- [ ] Add SQLite checkpointer so runs survive crashes
- [ ] Timeout handling for developer nodes
- [ ] Retry logic with exponential backoff for LLM calls
- [ ] Worktree cleanup on process exit (signal handlers)

### Phase 3: Observability
- [ ] LangSmith tracing integration
- [ ] Structured logging with run IDs
- [ ] Terminal UI with progress bars for parallel developers
- [ ] Cost tracking per run

### Phase 4: Advanced Patterns
- [ ] Configurable number of developers per sub-task
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

**Why stubs first?**
The hard part isn't calling an LLM — it's the orchestration topology, state management, and control flow. Getting the graph structure right with stubs means we can wire in any LLM provider without touching the core architecture.

## License

MIT
