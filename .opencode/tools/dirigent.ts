import { tool } from "@opencode-ai/plugin"
import path from "path"
import fs from "fs"

const DIRIGENT_BIN = path.join(
  import.meta.dir,
  "..",
  "..",
  "bin",
  "dirigent",
)

const GO_DIR = path.join(import.meta.dir, "..", "..", "go")

async function ensureBinary(): Promise<string> {
  if (fs.existsSync(DIRIGENT_BIN)) {
    return DIRIGENT_BIN
  }
  // Auto-build if Go source exists
  if (fs.existsSync(path.join(GO_DIR, "go.mod"))) {
    const binDir = path.dirname(DIRIGENT_BIN)
    if (!fs.existsSync(binDir)) {
      fs.mkdirSync(binDir, { recursive: true })
    }
    await Bun.$`cd ${GO_DIR} && go build -o ${DIRIGENT_BIN} ./cmd/dirigent`
    if (fs.existsSync(DIRIGENT_BIN)) {
      return DIRIGENT_BIN
    }
  }
  throw new Error(
    `Dirigent binary not found at ${DIRIGENT_BIN}. Run 'make build' from the project root.`,
  )
}

export const init = tool({
  description:
    "Initialize a new Dirigent coordination session for parallel task execution",
  args: {
    objective: tool.schema
      .string()
      .describe("The high-level objective to decompose and build"),
  },
  async execute(args) {
    const bin = await ensureBinary()
    const result = await Bun.$`${bin} init ${args.objective}`.text()
    return result.trim()
  },
})

export const plan = tool({
  description:
    "Store a decomposed task plan into the active Dirigent session",
  args: {
    plan: tool.schema
      .string()
      .describe(
        'JSON string with tasks, shared_contracts, and execution_order. Example: {"tasks":[{"id":"auth","title":"Auth module","description":"Build OAuth","depends_on":[],"interfaces":["AuthService"]}],"shared_contracts":["AuthService interface"],"execution_order":["auth"]}',
      ),
  },
  async execute(args) {
    const bin = await ensureBinary()
    const result = await Bun.$`${bin} plan ${args.plan}`.text()
    return result.trim()
  },
})

export const status = tool({
  description:
    "Query the current Dirigent session status. Optionally filter by task ID to see a specific task's progress, interfaces, and files modified.",
  args: {
    taskId: tool.schema
      .string()
      .optional()
      .describe("Optional task ID to get status for a specific task"),
  },
  async execute(args) {
    const bin = await ensureBinary()
    const cmdArgs = args.taskId ? [args.taskId] : []
    const result = await Bun.$`${bin} status ${cmdArgs}`.text()
    return result.trim()
  },
})

export const update = tool({
  description:
    "Update a task's status and artifacts in the Dirigent session. Use this to mark tasks as in_progress, completed, or failed, and to record which files were modified and what interfaces were exposed.",
  args: {
    taskId: tool.schema.string().describe("The task ID to update"),
    update: tool.schema
      .string()
      .describe(
        'JSON string with update fields. Example: {"status":"completed","result":"Built auth module","files_modified":["auth.go"],"interfaces":["AuthService"]}',
      ),
  },
  async execute(args) {
    const bin = await ensureBinary()
    const result =
      await Bun.$`${bin} update ${args.taskId} ${args.update}`.text()
    return result.trim()
  },
})

export const contracts = tool({
  description:
    "List or add shared interface contracts that coordinate between parallel tasks. Without arguments, lists all contracts. With a JSON argument, adds a new contract.",
  args: {
    contract: tool.schema
      .string()
      .optional()
      .describe(
        'Optional JSON string to add a contract. Example: {"name":"AuthService","owner":"auth-task","description":"Authentication interface","definition":"type AuthService interface { Login(credentials Credentials) (Token, error) }"}',
      ),
  },
  async execute(args) {
    const bin = await ensureBinary()
    const cmdArgs = args.contract ? [args.contract] : []
    const result = await Bun.$`${bin} contracts ${cmdArgs}`.text()
    return result.trim()
  },
})

export const ready = tool({
  description:
    "List tasks whose dependencies are all completed and are ready to be started. Use this to determine which tasks to fan out next.",
  args: {},
  async execute() {
    const bin = await ensureBinary()
    const result = await Bun.$`${bin} ready`.text()
    return result.trim()
  },
})

export const cleanup = tool({
  description:
    "Remove the Dirigent session file after the coordination is complete.",
  args: {},
  async execute() {
    const bin = await ensureBinary()
    const result = await Bun.$`${bin} cleanup`.text()
    return result.trim()
  },
})
