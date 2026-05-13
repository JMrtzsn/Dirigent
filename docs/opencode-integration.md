# OpenCode Integration

Dirigent integrates with OpenCode as a coordination layer for parallel agent work. It decomposes features into tasks, tracks dependencies, manages shared contracts between agents, and ensures parallel work merges cleanly.

## Prerequisites

- [OpenCode](https://opencode.ai) installed
- Go 1.21+ (to compile the binary)

## Setup

```bash
# Clone the repo
git clone https://github.com/johannesmauritzson/dirigent.git
cd dirigent

# Build the Go binary
make build
```

That's it. OpenCode auto-discovers the extensions from `.opencode/` when you open the project.

## Architecture

Dirigent uses three OpenCode extension points:

| File | Extension Type | What it does |
|---|---|---|
| `.opencode/tools/dirigent.ts` | Custom Tools | 7 tools the LLM can call during conversations |
| `.opencode/plugins/dirigent.ts` | Plugin | Registers `/dirigent` command + preserves state on compaction |
| `.opencode/skills/dirigent/SKILL.md` | Agent Skill | Coordination protocol loaded by sub-agents on demand |

### How they connect

```
User: /dirigent "Build feature X"
  │
  ├─ Plugin (config hook) registered the /dirigent command
  │
  ├─ LLM receives the orchestration prompt
  │
  ├─ LLM calls dirigent_init, dirigent_plan, dirigent_contracts
  │   └─ Custom tools invoke the Go binary → state written to .dirigent-session.json
  │
  ├─ LLM calls dirigent_ready → gets parallelizable tasks
  │
  ├─ LLM spawns Task sub-agents (OpenCode's built-in parallel execution)
  │   ├─ Sub-agents load the dirigent skill for coordination protocol
  │   ├─ Sub-agents call dirigent_update (mark in_progress / completed)
  │   └─ Sub-agents call dirigent_status / dirigent_contracts to coordinate
  │
  ├─ Orchestrator calls dirigent_ready for next wave (dependency-aware)
  │
  └─ All done → dirigent_cleanup
```

### State Management

The Go binary (`bin/dirigent`) manages a JSON session file (`.dirigent-session.json`) in the project root. This file tracks:

- The objective
- All tasks with their status, dependencies, and results
- Shared interface contracts
- Which files each task modified

## Tools Reference

All tools are defined in `.opencode/tools/dirigent.ts`. Each is a named export that becomes a tool with the naming pattern `dirigent_<export-name>`.

| Tool | Purpose | Example |
|---|---|---|
| `dirigent_init` | Create a new coordination session | `dirigent_init(objective: "Build auth module")` |
| `dirigent_plan` | Store the task decomposition | `dirigent_plan(plan: '{"tasks":[...],...}')` |
| `dirigent_status` | Query session or specific task | `dirigent_status()` or `dirigent_status(taskId: "auth")` |
| `dirigent_update` | Update task status/artifacts | `dirigent_update(taskId: "auth", update: '{"status":"completed"}')` |
| `dirigent_contracts` | List or add shared interfaces | `dirigent_contracts(contract: '{"name":"AuthService",...}')` |
| `dirigent_ready` | Get tasks with all deps completed | `dirigent_ready()` |
| `dirigent_cleanup` | Remove session file when done | `dirigent_cleanup()` |

### How Custom Tools Work

Custom tools in OpenCode are TypeScript files in `.opencode/tools/`. Each named export becomes a callable tool. The file uses the `@opencode-ai/plugin` package for type-safe tool definitions with Zod schemas.

Our tools are thin wrappers that:

1. Locate the compiled Go binary (`bin/dirigent`)
2. Spawn it as a subprocess with the appropriate arguments
3. Return the JSON output to the LLM

```typescript
export const init = tool({
  description: "Initialize a new Dirigent coordination session",
  args: {
    objective: tool.schema.string().describe("The objective"),
  },
  async execute(args) {
    return run(["init", args.objective])
  },
})
```

Dependencies for custom tools go in `.opencode/package.json`:

```json
{
  "dependencies": {
    "@opencode-ai/plugin": "1.14.41"
  }
}
```

OpenCode runs `bun install` at startup to install these.

## Plugin

The plugin (`.opencode/plugins/dirigent.ts`) does two things:

### 1. Command Registration

Uses the `config` hook to inject `/dirigent` as a slash command at runtime:

```typescript
async config(config) {
  config.command = config.command ?? {}
  config.command["dirigent"] = {
    description: "Decompose a feature into parallel tasks...",
    template: `...orchestration prompt...`,
  }
}
```

### 2. Compaction Hook

When OpenCode compacts context (to stay within token limits), the plugin injects the current coordination state so agents don't lose track of the session:

```typescript
"experimental.session.compacting": async (_input, output) => {
  const status = await run(["status"])
  output.context.push(`## Dirigent Coordination State\n${status}`)
}
```

## Skill

The skill (`.opencode/skills/dirigent/SKILL.md`) defines the coordination protocol for sub-agents. When a sub-agent loads this skill, it learns:

- How to report status
- How to read and respect shared contracts
- How to check on sibling agents
- File ownership rules (don't modify files owned by other tasks)

## Usage

```
/dirigent "Add user authentication with OAuth, session management, and rate limiting"
```

The orchestrator will:

1. Analyze the codebase
2. Decompose into tasks (e.g., "oauth", "sessions", "rate-limiter")
3. Define shared contracts (e.g., `AuthToken` type, `UserSession` interface)
4. Fan out independent tasks in parallel
5. Wait for dependent tasks (e.g., sessions depends on auth)
6. Verify and merge results

## Go Binary (CLI)

The coordination state is managed by a Go binary with 7 subcommands:

```
dirigent init <objective>          # Create session
dirigent plan '<json>'             # Store task plan
dirigent status [task-id]          # Query state
dirigent update <id> '<json>'      # Update task
dirigent contracts ['<json>']      # List/add contracts
dirigent ready                     # Tasks with deps met
dirigent cleanup                   # Remove session file
```

All output is JSON for machine consumption.

Source: `go/cmd/dirigent/`

## Project Structure

```
.opencode/
  tools/dirigent.ts          # Custom tools (7 tools)
  plugins/dirigent.ts        # Plugin (command + compaction hook)
  skills/dirigent/SKILL.md   # Coordination protocol
  package.json               # @opencode-ai/plugin dependency

go/
  cmd/dirigent/              # CLI binary source
    main.go                  # Subcommand routing
    state/state.go           # Session persistence
    state/state_test.go      # Tests
  core/                      # Agent framework (interface, registry, messages)
  patterns/                  # Delegate, Fanout, ReviewLoop
  runtime/                   # Execution engine

plugin/                      # npm-publishable package (future distribution)
  src/index.ts
  commands/dirigent.md
  scripts/install-binary.mjs
  package.json

bin/dirigent                 # Compiled binary (gitignored, built via `make build`)
```

## Development

```bash
make build       # Build Go binary to bin/ and plugin/bin/
make go-test     # Run Go tests
make go-lint     # Run go vet
make check-all   # All checks (Python + Go)
```

## Distribution (Future)

The `plugin/` directory contains an npm-publishable package. Once published:

```json
// opencode.json
{
  "plugin": ["opencode-dirigent"]
}
```

The postinstall script builds the Go binary and copies the `/dirigent` command to `~/.config/opencode/commands/`.
