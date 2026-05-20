# Dirigent Go Framework

Parallel agent coordination engine — the CLI binary behind the OpenCode plugin.

## Architecture

| Package | Purpose |
|---------|---------|
| `core` | Agent interface, capability registry, request/response messages |
| `runtime` | Execution engine — routes delegations with timeout and depth limits |
| `patterns` | Composable patterns: Delegate, Fanout, ReviewLoop |
| `cmd/dirigent` | CLI binary (cobra) — JSON session management |
| `cmd/dirigent/state` | Session persistence (`.dirigent-session.json`) |
| `utils` | Repository scanning (file tree, key files) |

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
// core.Agent — interface all agents implement
type Agent interface {
    Info() AgentInfo
    Handle(ctx context.Context, req DelegateRequest) (DelegateResponse, error)
}

// state.Session — coordination state (objective, plan, contracts)
// state.Task — unit of work (status, depends_on, branch, completed_at)
```

## Patterns

| Pattern | Description |
|---------|-------------|
| `Delegate` | Send a task to a single agent by name or capability |
| `Fanout` | Send the same task to N agents in parallel |
| `ReviewLoop` | Produce → review → revise with max rounds |
