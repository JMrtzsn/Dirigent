import { tool } from "@opencode-ai/plugin"
import path from "path"
import fs from "fs"

const worktree = process.cwd()
const bin = path.join(worktree, "bin", "dirigent")
const altBin = path.join(worktree, "plugin", "bin", "dirigent")

function getBin(): string {
  if (fs.existsSync(bin)) return bin
  if (fs.existsSync(altBin)) return altBin
  throw new Error(`Dirigent binary not found. Run 'make build'`)
}

async function run(args: string[]): Promise<string> {
  const b = getBin()
  const proc = Bun.spawn([b, ...args], { cwd: worktree, stdout: "pipe", stderr: "pipe" })
  const text = await new Response(proc.stdout).text()
  await proc.exited
  return text.trim()
}

export const init = tool({
  description: "Initialize a new Dirigent coordination session for parallel task execution",
  args: {
    objective: tool.schema.string().describe("The high-level objective to decompose and build"),
  },
  async execute(args) {
    return run(["init", args.objective])
  },
})

export const plan = tool({
  description: "Store a decomposed task plan into the active Dirigent session",
  args: {
    plan: tool.schema.string().describe("JSON string with tasks, shared_contracts, and execution_order"),
  },
  async execute(args) {
    return run(["plan", args.plan])
  },
})

export const status = tool({
  description: "Query the current Dirigent session status. Optionally filter by task ID.",
  args: {
    taskId: tool.schema.string().optional().describe("Optional task ID to query a specific task"),
  },
  async execute(args) {
    return args.taskId ? run(["status", args.taskId]) : run(["status"])
  },
})

export const update = tool({
  description: "Update a task's status and artifacts. Use to mark tasks in_progress, completed, or failed.",
  args: {
    taskId: tool.schema.string().describe("The task ID to update"),
    update: tool.schema.string().describe('JSON string: {"status":"completed","result":"...","files_modified":[...],"interfaces":[...]}'),
  },
  async execute(args) {
    return run(["update", args.taskId, args.update])
  },
})

export const contracts = tool({
  description: "List or add shared interface contracts between parallel tasks.",
  args: {
    contract: tool.schema.string().optional().describe("Optional JSON to add a contract"),
  },
  async execute(args) {
    return args.contract ? run(["contracts", args.contract]) : run(["contracts"])
  },
})

export const ready = tool({
  description: "List tasks whose dependencies are all completed — ready to be started.",
  args: {},
  async execute() {
    return run(["ready"])
  },
})

export const cleanup = tool({
  description: "Remove the Dirigent session file after completion.",
  args: {},
  async execute() {
    return run(["cleanup"])
  },
})
