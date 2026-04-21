# Dirigent

Fan-out/fan-in AI agent orchestrator built with [LangGraph](https://langchain-ai.github.io/langgraph/). Decomposes tasks into small PRs, spawns parallel developer agents in isolated git worktrees, reviews and selects the best implementation. Delivers a draft PR.

## How It Works

Dirigent uses a Short-Lived Integration Branch (SLIB) workflow. Each sub-task is handled sequentially so later work builds on top of earlier approved changes.

1. **Architect** breaks a task into ordered, atomic sub-tasks
2. **Feature branch** is created off `main` (e.g. `feature/refactor-auth-module`)
3. For each sub-task:
   - **Fan Out** — N developer agents work it in parallel, each in its own git worktree branching off the feature branch
   - **Fan In** — Reviewer scores all implementations, selects the best
   - **Human Review** — you inspect and approve before anything is merged
   - **Merge** — winning branch is merged into the feature branch
4. After all sub-tasks are merged, Dirigent pushes the feature branch and opens a **draft PR** to `main`

The human review gate between each sub-task is the bottleneck by design — nothing advances until you approve.

## Setup

```bash
git clone <repo-url> && cd Dirigent
make dev          # venv + deps
make check        # lint + tests
```

Requires a GitHub Copilot subscription. Token is auto-resolved from OpenCode's auth cache (`~/.local/share/opencode/auth.json`) or from the `DIRIGENT_COPILOT_TOKEN` env var.

## Usage

```bash
dirigent "Refactor the auth module" --repo /path/to/repo
```

The graph pauses at each human review step. You'll see the reviewer's recommendation and can approve (`yes`) or reject (`no`) to retry that sub-task. After the last sub-task is approved and merged, Dirigent pushes the feature branch and creates a draft PR.

```
dirigent <objective> [options]

  --repo PATH            Target git repository (default: .)
  --developers N         Parallel developer agents (default: 3)
  -v, --verbose          Debug logging
```

## License

MIT
