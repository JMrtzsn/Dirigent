---
name: dirigent
description: Coordinate parallel agent tasks with shared contracts, dependency tracking, and status synchronization for building complex features
---

## Dirigent Coordination Protocol

You are participating in a coordinated multi-agent build managed by Dirigent. Follow this protocol.

### As the Orchestrator (when running /dirigent)

1. **Decompose** the objective into independent tasks with clear boundaries
2. **Identify shared contracts** — interfaces, types, or APIs that multiple tasks depend on. Define these BEFORE spawning agents.
3. **Map dependencies** — which tasks block others
4. **Assign file ownership** — each task should have exclusive files to avoid merge conflicts
5. **Fan out** — launch independent tasks in parallel via the Task tool
6. **Monitor** — check `dirigent status` periodically
7. **Merge** — once all tasks complete, review and integrate

### As a Sub-Agent (when spawned by the orchestrator)

1. **Read your assignment** — understand your task, dependencies, and contracts
2. **Mark in-progress** — call `dirigent update <your-task-id> '{"status": "in_progress"}'`
3. **Check contracts** — call `dirigent contracts` to see shared interfaces you must implement or consume
4. **Check siblings** — call `dirigent status` if you need to know what other agents are doing
5. **Stay in your lane** — only modify files assigned to you
6. **Report completion** — call `dirigent update <your-task-id> '{"status": "completed", "result": "...", "files_modified": [...], "interfaces": [...]}'`

### Contract Rules

- Contracts are agreed-on interfaces between tasks
- The **owner** task defines and implements the contract
- Other tasks that depend on the contract must code against it
- If you need a contract that doesn't exist, flag it — don't invent your own

### Conflict Avoidance

- Each task has assigned files — respect boundaries
- Shared packages/modules should be owned by exactly one task
- If you discover you need to modify a file owned by another task, check `dirigent status` and coordinate
