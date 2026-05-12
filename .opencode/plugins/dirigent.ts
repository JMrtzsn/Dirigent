import type { Plugin } from "@opencode-ai/plugin"
import { tool } from "@opencode-ai/plugin"
import path from "path"
import fs from "fs"

export const DirigentPlugin: Plugin = async (ctx) => {
  const bin = path.join(ctx.worktree, "bin", "dirigent")
  const altBin = path.join(ctx.worktree, "plugin", "bin", "dirigent")
  const cwd = ctx.worktree

  function getBin(): string {
    if (fs.existsSync(bin)) return bin
    if (fs.existsSync(altBin)) return altBin
    throw new Error(`Dirigent binary not found. Run 'make build' from ${cwd}`)
  }

  async function run(args: string[]): Promise<string> {
    const b = getBin()
    const proc = Bun.spawn([b, ...args], { cwd, stdout: "pipe", stderr: "pipe" })
    const text = await new Response(proc.stdout).text()
    await proc.exited
    return text.trim()
  }

  return {
    tool: {
      dirigent_init: tool({
        description: "Initialize a new Dirigent coordination session for parallel task execution",
        args: {
          objective: tool.schema.string().describe("The high-level objective to decompose and build"),
        },
        async execute(args) {
          return run(["init", args.objective])
        },
      }),

      dirigent_plan: tool({
        description: "Store a decomposed task plan into the active Dirigent session",
        args: {
          plan: tool.schema.string().describe("JSON string with tasks, shared_contracts, and execution_order"),
        },
        async execute(args) {
          return run(["plan", args.plan])
        },
      }),

      dirigent_status: tool({
        description: "Query the current Dirigent session status. Optionally filter by task ID.",
        args: {
          taskId: tool.schema.string().optional().describe("Optional task ID to query a specific task"),
        },
        async execute(args) {
          return args.taskId ? run(["status", args.taskId]) : run(["status"])
        },
      }),

      dirigent_update: tool({
        description: "Update a task's status and artifacts. Use to mark tasks in_progress, completed, or failed.",
        args: {
          taskId: tool.schema.string().describe("The task ID to update"),
          update: tool.schema.string().describe('JSON string: {"status":"completed","result":"...","files_modified":[...],"interfaces":[...]}'),
        },
        async execute(args) {
          return run(["update", args.taskId, args.update])
        },
      }),

      dirigent_contracts: tool({
        description: "List or add shared interface contracts between parallel tasks.",
        args: {
          contract: tool.schema.string().optional().describe('Optional JSON to add a contract'),
        },
        async execute(args) {
          return args.contract ? run(["contracts", args.contract]) : run(["contracts"])
        },
      }),

      dirigent_ready: tool({
        description: "List tasks whose dependencies are all completed — ready to be started.",
        args: {},
        async execute() {
          return run(["ready"])
        },
      }),

      dirigent_cleanup: tool({
        description: "Remove the Dirigent session file after completion.",
        args: {},
        async execute() {
          return run(["cleanup"])
        },
      }),
    },

    async config(config) {
      config.command = config.command ?? {}
      config.command["dirigent"] = {
        description: "Decompose a feature into parallel tasks, coordinate agents, and merge results",
        template: `You are the Dirigent orchestrator. Decompose the user's objective into parallel tasks, coordinate sub-agents, and merge results.

**Objective:** $ARGUMENTS

## Protocol

### 1. Analyze the Codebase
Read key files (README, config, entry points) to understand the project structure, language, framework, and patterns.

### 2. Decompose into Tasks
Split the objective into parallel tasks. For each task define:
- Clear file/module boundary (what it owns exclusively)
- Interfaces it exposes or consumes
- Dependencies on other tasks
- Maximize parallelism

### 3. Initialize and Store Plan
Call dirigent_init, then dirigent_plan, then dirigent_contracts for each shared interface.

Plan JSON structure:
{"tasks":[{"id":"...","title":"...","description":"...","depends_on":[],"interfaces":[]}],"shared_contracts":["..."],"execution_order":["..."]}

Define ALL contracts BEFORE spawning agents.

### 4. Fan Out (Dependency-Aware Waves)
Call dirigent_ready to get startable tasks. For each, spawn a Task sub-agent with:
- Task title and full description
- Files it owns (and must NOT touch)
- Shared contracts to implement/consume
- Instructions to call dirigent_update when starting and finishing
- Instructions to call dirigent_status and dirigent_contracts for coordination

Launch all ready tasks in parallel. After each wave completes, call dirigent_ready again.

### 5. Verify and Finalize
1. dirigent_status — confirm all tasks completed
2. Review combined changes for contract consistency
3. Run build/tests
4. Present summary to user
5. dirigent_cleanup`,
      }
    },

    "experimental.session.compacting": async (_input, output) => {
      try {
        const status = await run(["status"])
        if (status && !status.includes("error")) {
          output.context.push(`## Dirigent Coordination State\n\n\`\`\`json\n${status}\n\`\`\`\n\nUse dirigent_status, dirigent_ready, dirigent_update tools to continue.`)
        }
      } catch {
        // No active session
      }
    },
  }
}
