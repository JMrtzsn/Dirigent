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
| `diff <task-id>` | Show git diff of task branch vs base |
| `review <task-id>` | Show task details, diff stats, open worktree |
| `verify <task-id>` | Human-approve a task (unlocks merge) |
| `merge <task-id>` | Merge task branch back to base, remove worktree |
| `log` | Show completed tasks with timestamps |
| `status [task-id]` | Query session or task status |
| `update <task-id> '<json>'` | Update task status/result |
| `contracts [json]` | Manage shared interface contracts |
| `list-worktrees` | Show active task worktrees |
| `cleanup` | Remove all worktrees, branches, and session |

All output is JSON (except `graph` without `--json`).

## Flow

```
init → plan → checkout (parallel) → agents work → update → verify → merge → cleanup
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

Dirigent runs as an OpenCode plugin via `.opencode/tools/dirigent.ts`. The orchestrator (OpenCode's main agent) calls Dirigent tools to coordinate sub-agents spawned via the Task tool.

```
.opencode/tools/dirigent.ts      — 10 custom tools calling the Go binary
.opencode/plugins/dirigent.ts    — /dirigent command + compaction hook
.opencode/skills/dirigent/       — coordination protocol for the LLM
```

## Build

```sh
make build
# or: cd go && go build -o ../bin/dirigent ./cmd/dirigent
```

## Test

```sh
make go-test
# or: cd go && go test ./...
```

## Architecture

```
go/
├── cmd/dirigent/         CLI binary (cobra)
│   └── state/            Session persistence (.dirigent-session.json)
├── core/                 Agent interface, messages, capability registry
├── runtime/              Execution engine (timeout, traces, depth limit)
├── patterns/             Delegate, Fanout, ReviewLoop
└── utils/                Repo scanning (file tree, key files)
```

## License

MIT
