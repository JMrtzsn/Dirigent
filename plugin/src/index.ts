import { type Plugin, tool } from "@opencode-ai/plugin"
import { resolve, dirname } from "path"
import { existsSync } from "fs"
import { fileURLToPath } from "url"

const __dirname = dirname(fileURLToPath(import.meta.url))
const BIN = resolve(__dirname, "..", "bin", "dirigent")

async function run(...args: string[]): Promise<string> {
  if (!existsSync(BIN)) {
    throw new Error(
      `Dirigent binary not found at ${BIN}. Run 'npm run build:go' in the opencode-dirigent plugin directory.`,
    )
  }
  const result = await Bun.$`${BIN} ${args}`.text()
  return result.trim()
}

export const DirigentPlugin: Plugin = async ({ client, $ }) => {
  return {
    // --- Custom tools the LLM can call ---
    tool: {
      dirigent_init: tool({
        description:
          "Initialize a new Dirigent coordination session for parallel task execution",
        args: {
          objective: tool.schema
            .string()
            .describe("The high-level objective to decompose and build"),
        },
        async execute(args) {
          return run("init", args.objective)
        },
      }),

      dirigent_plan: tool({
        description:
          "Store a decomposed task plan into the active Dirigent session",
        args: {
          plan: tool.schema
            .string()
            .describe(
              'JSON string with tasks, shared_contracts, and execution_order',
            ),
        },
        async execute(args) {
          return run("plan", args.plan)
        },
      }),

      dirigent_status: tool({
        description:
          "Query the current Dirigent session status. Optionally filter by task ID.",
        args: {
          taskId: tool.schema
            .string()
            .optional()
            .describe("Optional task ID to query a specific task"),
        },
        async execute(args) {
          return args.taskId ? run("status", args.taskId) : run("status")
        },
      }),

      dirigent_update: tool({
        description:
          "Update a task's status and artifacts. Use to mark tasks in_progress, completed, or failed.",
        args: {
          taskId: tool.schema.string().describe("The task ID to update"),
          update: tool.schema
            .string()
            .describe(
              'JSON string: {"status":"completed","result":"...","files_modified":[...],"interfaces":[...]}',
            ),
        },
        async execute(args) {
          return run("update", args.taskId, args.update)
        },
      }),

      dirigent_contracts: tool({
        description:
          "List or add shared interface contracts between parallel tasks.",
        args: {
          contract: tool.schema
            .string()
            .optional()
            .describe(
              'Optional JSON to add: {"name":"...","owner":"task-id","description":"...","definition":"..."}',
            ),
        },
        async execute(args) {
          return args.contract
            ? run("contracts", args.contract)
            : run("contracts")
        },
      }),

      dirigent_ready: tool({
        description:
          "List tasks whose dependencies are all completed — ready to be started.",
        args: {},
        async execute() {
          return run("ready")
        },
      }),

      dirigent_cleanup: tool({
        description: "Remove the Dirigent session file after completion.",
        args: {},
        async execute() {
          return run("cleanup")
        },
      }),
    },

    // --- Preserve coordination state across compaction ---
    "experimental.session.compacting": async (_input, output) => {
      try {
        const status = await run("status")
        if (status && !status.includes("error")) {
          output.context.push(`## Dirigent Coordination State

There is an active Dirigent coordination session. Here is the current state:

\`\`\`json
${status}
\`\`\`

Use the dirigent_status, dirigent_ready, and dirigent_update tools to continue coordinating tasks.`)
        }
      } catch {
        // No active session — nothing to preserve
      }
    },
  }
}
