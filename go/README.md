# Dirigent Go Framework — parallel agent coordination engine

## Architecture

| Package | Purpose |
|---------|---------|
| `core` | Fundamental types: Agent interface, capability registry, request/response messages |
| `runtime` | Execution engine — routes delegations to agents via capability/name lookup with timeout and depth limits |
| `patterns` | Composable interaction patterns: Delegate, Fanout, ReviewLoop |
| `cmd/dirigent` | CLI binary — JSON-based session management for coordinating parallel tasks |
| `cmd/dirigent/state` | Session persistence — plans, tasks, contracts stored in `.dirigent-session.json` |
| `utils` | Repository scanning helpers (file tree, key file reading) |

## Key Types

```go
// core.Agent — interface all agents implement
type Agent interface {
    Info() AgentInfo
    Handle(ctx context.Context, req DelegateRequest) (DelegateResponse, error)
}

// core.DelegateRequest / DelegateResponse — agent-to-agent messages
// runtime.Runtime — orchestrates agent execution with tracing
// state.Session — top-level coordination state (ID, objective, plan, contracts)
// state.Plan — decomposed tasks with dependency graph
// state.Task — unit of work (status, depends_on, files_modified, result)
```

## Key Patterns

| Pattern | Description |
|---------|-------------|
| `Delegate` | Send a task to a single agent by name or capability |
| `Fanout` | Send the same task to multiple agents concurrently, collect all results |
| `ReviewLoop` | Produce → review → revise cycle with configurable max rounds |

## Build

```sh
make build
# or directly:
cd go && go build -o ../bin/dirigent ./cmd/dirigent
```

## Test

```sh
make go-test
# or directly:
cd go && go test ./...
```

## Lint

```sh
make go-lint
# or: cd go && go vet ./...
```

## CLI Usage

```
dirigent --help
```

Subcommands:

| Command | Description |
|---------|-------------|
| `init <objective>` | Initialize a new coordination session |
| `plan '<json>'` | Store a decomposed task plan |
| `status [task-id]` | Query task or session status |
| `update <task-id> '<json>'` | Update a task's status and artifacts |
| `contracts [json]` | List or add shared interface contracts |
| `ready` | List tasks whose dependencies are all completed |
| `cleanup` | Remove the session file |

All output is JSON for machine consumption.
