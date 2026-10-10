# Dirigent

Parallel agent coordination for OpenCode. Decomposes features into tasks, fans out to sub-agents in isolated git worktrees, tracks dependencies, and merges results.

## What it does

```
dirigent init "build feature X" --require-verification
dirigent plan '{"tasks":[...]}'
dirigent graph

Wave 1
  [ ] task-a — Backend API
  [ ] task-b — Frontend UI
  |
  v
Wave 2 (after: task-a, task-b)
  [ ] task-c — Integration tests
```

Each task gets its own git worktree — true parallel isolation. No branch conflicts.

## Commands

| Command | Description |
|---------|-------------|
| `init <objective>` | Start a coordination session |
| `plan '<json>'` | Store task decomposition with dependencies |
| `graph` | Visualize dependency waves (with color, or `--json`) |
| `ready` | List tasks whose deps are satisfied |
| `checkout <task-id>` | Create worktree + branch for a task |
| `spawn <task-id> [--model m] [--agent a]` | Check out a task and run a separate headless `opencode` in its worktree |
| `diff <task-id>` | Show git diff of task branch vs base |
| `review <task-id>` | Show task details, diff stats, open worktree |
| `verify <task-id>` | Human-approve a task (unlocks merge) |
| `merge <task-id>` | Commit uncommitted worker changes, merge task branch into base, remove worktree |
| `log` | Show completed tasks with timestamps |
| `status [task-id]` | Query session or task status |
| `update <task-id> '<json>'` | Update task status/result |
| `contracts [json]` | Manage shared interface contracts |
| `list-worktrees` | Show active task worktrees |
| `cleanup` | Remove all worktrees, branches, and session |

All output is JSON (except `graph` without `--json`).

## Flow

```
init → plan → spawn (parallel) → workers run → update → verify → merge → cleanup
```

With `--require-verification`:
```
completed → [blocked from merge] → verify → merge
```

Without it:
```
completed → merge
```

## Integration with OpenCode

`make install` links the binary to `~/.local/bin/dirigent` and copies
`plugin/dirigent.ts` into `~/.config/opencode/plugins/`, so it works in any repo.
The plugin is copied, not symlinked, so it resolves `@opencode-ai/plugin` from
the global config. Rerun `make install` after changing it.

Run `/dirigent <objective>` in opencode. The orchestrator plans tasks, then
`dirigent_spawn` starts one `opencode run` process per task inside its worktree,
so workers never share a working directory. Workers find the session through
`DIRIGENT_SESSION` and report with `dirigent update`. `dirigent_wait` blocks
until all workers exit and marks silent exits as failed. Worker output goes to
`.dirigent-worktrees/<task>.log`.

Dirigent adds `.dirigent-worktrees/` and `.dirigent-session.json` to the host
repo's `.git/info/exclude`, never to its `.gitignore`.

## Build

```sh
make build
# or: cd go && go build -o ../bin/dirigent ./cmd/dirigent
```

## Test

```sh
make test
# or: cd go && go test ./...
```

## Architecture

```
go/
└── cmd/dirigent/         CLI binary (cobra)
    └── state/            Session persistence (.dirigent-session.json)
```

## License

MIT
