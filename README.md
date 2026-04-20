# Dirigent

Fan-out/fan-in AI agent orchestrator built with [LangGraph](https://langchain-ai.github.io/langgraph/). Decomposes tasks into small PRs, spawns parallel developer agents in isolated git worktrees, reviews and selects the best implementation.

## How It Works

1. **Architect** breaks a task into atomic PRs
2. **Fan Out** — N developer agents work the same sub-task in parallel, each in its own git worktree
3. **Fan In** — Reviewer tests, compares, selects the best implementation
4. **Human Stop** — you verify before anything gets committed
5. Repeat for the next PR

## Quickstart

```bash
git clone <repo-url> && cd Dirigent
make dev          # venv + deps
make check        # lint + tests

# Stub mode (no LLM, no API token)
dirigent "Refactor the auth module" --repo /path/to/repo --no-llm

# With LLM (needs GitHub Copilot subscription)
# Token auto-resolved from OpenCode auth cache or DIRIGENT_COPILOT_TOKEN env var
dirigent "Refactor the auth module" --repo /path/to/repo --developers 3 -v
```

### CLI

```
dirigent <objective> [options]

  --repo PATH            Target git repository (default: .)
  --developers N         Parallel developer agents (default: 3)
  --no-llm               Stub mode, no LLM calls
  --db PATH              SQLite checkpoint DB (default: .dirigent/checkpoints.db)
  -v, --verbose          Debug logging
```

## License

MIT
