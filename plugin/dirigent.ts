import type { Plugin } from "@opencode-ai/plugin"
import { tool } from "@opencode-ai/plugin"

type Task = { id: string; status: string; pid?: number; log?: string }

export const DirigentPlugin: Plugin = async (ctx) => {
  const cwd = ctx.worktree

  async function run(args: string[]): Promise<string> {
    const proc = Bun.spawn(["dirigent", ...args], { cwd, stdout: "pipe", stderr: "pipe" })
    const [out, err, code] = await Promise.all([
      new Response(proc.stdout).text(),
      new Response(proc.stderr).text(),
      proc.exited,
    ])
    if (code !== 0) throw new Error(err.trim() || `dirigent ${args[0]} exited ${code}`)
    return out.trim()
  }

  function alive(pid: number): boolean {
    try {
      process.kill(pid, 0)
      return true
    } catch {
      return false
    }
  }

  async function workers(): Promise<Task[]> {
    const session = JSON.parse(await run(["status"]))
    return (session.plan?.tasks ?? []).filter((t: Task) => t.pid)
  }

  const taskId = tool.schema.string().describe("Task ID from the plan")

  return {
    tool: {
      dirigent_init: tool({
        description: "Start a Dirigent session in the current repo",
        args: {
          objective: tool.schema.string().describe("High-level objective"),
          requireVerification: tool.schema.boolean().optional().describe("Require human `dirigent verify` before merge"),
        },
        async execute(args) {
          return run(["init", args.objective, ...(args.requireVerification ? ["--require-verification"] : [])])
        },
      }),

      dirigent_plan: tool({
        description: 'Store the task plan. JSON: {"tasks":[{"id","title","description","depends_on":[],"interfaces":[]}]}',
        args: { plan: tool.schema.string().describe("Plan JSON") },
        async execute(args) {
          return run(["plan", args.plan])
        },
      }),

      dirigent_contracts: tool({
        description: 'List shared contracts, or add one: {"name","owner","description","definition"}',
        args: { contract: tool.schema.string().optional() },
        async execute(args) {
          return run(args.contract ? ["contracts", args.contract] : ["contracts"])
        },
      }),

      dirigent_status: tool({
        description: "Session or task status",
        args: { taskId: taskId.optional() },
        async execute(args) {
          return run(args.taskId ? ["status", args.taskId] : ["status"])
        },
      }),

      dirigent_ready: tool({
        description: "Pending tasks whose dependencies are completed",
        args: {},
        async execute() {
          return run(["ready"])
        },
      }),

      dirigent_spawn: tool({
        description:
          "Create the task's git worktree and launch a separate headless opencode in it. Returns immediately; use dirigent_wait.",
        args: {
          taskId,
          model: tool.schema.string().optional().describe("provider/model for the worker"),
          agent: tool.schema.string().optional().describe("Agent for the worker"),
        },
        async execute(args) {
          return run([
            "spawn",
            args.taskId,
            ...(args.model ? ["--model", args.model] : []),
            ...(args.agent ? ["--agent", args.agent] : []),
          ])
        },
      }),

      dirigent_wait: tool({
        description:
          "Block until every spawned worker has exited. Workers that exited without reporting are marked failed. Returns final task states.",
        args: { timeoutMinutes: tool.schema.number().optional().describe("Default 60") },
        async execute(args, context) {
          const deadline = Date.now() + (args.timeoutMinutes ?? 60) * 60_000
          while (!context.abort.aborted && Date.now() < deadline) {
            const running = (await workers()).filter((t) => t.status === "in_progress" && alive(t.pid!))
            if (running.length === 0) break
            await Bun.sleep(5_000)
          }
          const tasks = await workers()
          for (const t of tasks) {
            if (t.status === "in_progress" && !alive(t.pid!)) {
              await run(["update", t.id, JSON.stringify({ status: "failed", result: `worker exited without reporting, see ${t.log}` })])
              t.status = "failed"
            }
          }
          return JSON.stringify(tasks.map(({ id, status, log }) => ({ id, status, log })), null, 2)
        },
      }),

      dirigent_diff: tool({
        description: "Diff of a task branch against the base branch",
        args: { taskId },
        async execute(args) {
          return run(["diff", args.taskId])
        },
      }),

      dirigent_merge: tool({
        description: "Commit the worker's changes, merge the task branch into base, and remove its worktree",
        args: { taskId },
        async execute(args) {
          return run(["merge", args.taskId])
        },
      }),

      dirigent_cleanup: tool({
        description: "Remove all task worktrees, task branches, and the session file",
        args: {},
        async execute() {
          return run(["cleanup"])
        },
      }),
    },

    async config(config) {
      config.command = config.command ?? {}
      config.command["dirigent"] = {
        description: "Split work into parallel tasks, each run by a separate opencode in its own git worktree",
        template: `You are the Dirigent orchestrator. Objective: $ARGUMENTS

1. Read enough of the repo to understand structure, language, and test commands.
2. Split the objective into tasks with exclusive file ownership and explicit depends_on. Maximize parallelism; do not split work that touches the same files.
3. dirigent_init, then dirigent_plan, then dirigent_contracts for every shared interface. Define all contracts before spawning.
4. Wave loop:
   a. dirigent_ready, then dirigent_spawn every ready task (each runs as a separate opencode in its own worktree).
   b. dirigent_wait.
   c. For each completed task: dirigent_diff, review it, then dirigent_merge. On merge conflict stop and report to the user.
   d. For failed tasks: read the log path, then either dirigent_spawn it again (it reuses the worktree and keeps partial work) or report to the user.
   e. Repeat until no tasks remain.
5. Run the project's build/tests on the merged result and fix integration issues.
6. Summarize to the user, then dirigent_cleanup. Never push or open PRs.`,
      }
    },

    "experimental.session.compacting": async (_input, output) => {
      try {
        output.context.push(`## Dirigent session\n\n\`\`\`json\n${await run(["status"])}\n\`\`\``)
      } catch {
        // no active session in this repo
      }
    },
  }
}
