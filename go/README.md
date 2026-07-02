# Dirigent Go CLI

Parallel agent coordination engine — a stateless CLI over a JSON session file.

## Architecture

| Package | Purpose |
|---------|---------|
| `cmd/dirigent` | CLI binary (cobra) — JSON session management + git worktree orchestration |
| `cmd/dirigent/state` | Session persistence (`.dirigent-session.json`) |

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

## CLI Commands

```
dirigent --help
```

| Command | Description |
|---------|-------------|
| `init <objective>` | Start a coordination session (`--require-verification`) |
| `plan '<json>'` | Store task decomposition with dependencies |
| `graph` | Visualize dependency waves (`--json`, `--no-color`) |
| `ready` | List tasks whose deps are satisfied |
| `status [task-id]` | Query session or task status |
| `update <task-id> '<json>'` | Update task status/result |
| `contracts [json]` | Manage shared interface contracts |
| `checkout <task-id>` | Create worktree + branch for a task |
| `diff <task-id>` | Show git diff of task branch vs base |
| `review <task-id>` | Show task details and diff stats |
| `verify <task-id>` | Human-approve a task (unlocks merge) |
| `merge <task-id>` | Merge task branch, remove worktree |
| `log` | Show completed tasks with timestamps |
| `list-worktrees` | Show active task worktrees |
| `cleanup` | Remove all worktrees, branches, session |

All output is JSON (except `graph` without `--json`).

## Key Types

```go
// state.Session — coordination state (objective, plan, contracts)
// state.Task — unit of work (status, depends_on, branch, completed_at)
```
