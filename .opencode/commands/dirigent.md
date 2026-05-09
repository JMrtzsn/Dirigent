---
description: Decompose a feature into parallel tasks, coordinate agents, and merge results
agent: build
subtask: false
---

You are now acting as the Dirigent orchestrator. The user wants to build a feature using parallel coordinated agents.

**Objective:** $ARGUMENTS

Load the `dirigent` skill first to understand the full coordination protocol.

Follow this protocol:

## Step 1: Analyze the Codebase

Before decomposing, understand the codebase:
- Read key files (README, config files, entry points) to understand the project structure
- Identify the language, framework, and patterns in use
- Understand how the existing code is organized (modules, packages, etc.)

## Step 2: Decompose into Tasks

Based on your codebase analysis, decompose the objective into parallel tasks. For each task, define:
- A clear boundary (which files/modules it owns)
- What interfaces it exposes or consumes
- Dependencies on other tasks
- Maximize parallelism — prefer more independent tasks over fewer dependent ones

## Step 3: Initialize Session

```
dirigent_init(objective: "$ARGUMENTS")
```

## Step 4: Store the Plan

Create the plan JSON and store it:
```
dirigent_plan(plan: '{"tasks":[...],"shared_contracts":[...],"execution_order":[...]}')
```

Each task needs: `id`, `title`, `description`, `depends_on` (array of task IDs), `interfaces` (array of interface names it exposes).

## Step 5: Define Contracts

For each shared interface between tasks, register it:
```
dirigent_contracts(contract: '{"name":"InterfaceName","owner":"task-id","description":"...","definition":"the actual type/interface definition"}')
```

Define ALL contracts BEFORE spawning any agents. This is critical — agents need to know the contracts upfront.

## Step 6: Fan Out (Dependency-Aware)

Use `dirigent_ready` to get tasks that can start now (all dependencies met):
```
dirigent_ready()
```

For each ready task, spawn a Task sub-agent with these instructions:

1. Your task: [task title and full description]
2. Files you own: [list of files this task should create/modify]
3. Files you must NOT touch: [files owned by other tasks]
4. Shared contracts you must implement: [relevant contract definitions]
5. Shared contracts you consume: [interfaces you depend on]
6. When you start, call: `dirigent_update(taskId: "<id>", update: '{"status":"in_progress"}')`
7. When you finish, call: `dirigent_update(taskId: "<id>", update: '{"status":"completed","result":"<summary>","files_modified":[...],"interfaces":[...]}')`
8. If you need to check what other agents are doing: `dirigent_status()`
9. If you need to see shared contracts: `dirigent_contracts()`

Launch all ready tasks in parallel using the Task tool.

## Step 7: Handle Dependencies

After the first batch completes, check for newly ready tasks:
```
dirigent_ready()
```

If there are more ready tasks, spawn them. Repeat until all tasks are done.

## Step 8: Collect and Verify

1. Call `dirigent_status()` to verify all tasks completed
2. Review the combined changes:
   - Do shared contracts match between producer and consumer?
   - Are there any file conflicts?
   - Does the code compile / pass tests?
3. Fix any integration issues

## Step 9: Finalize

1. Run the project's build/test commands to verify everything works
2. Present a summary to the user with what was built
3. Call `dirigent_cleanup()` to remove the session file
