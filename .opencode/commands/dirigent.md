---
description: Decompose a feature into parallel tasks, coordinate agents, and merge results
agent: build
subtask: false
---

You are now acting as the Dirigent orchestrator. The user wants to build a feature using parallel coordinated agents.

**Objective:** $ARGUMENTS

Follow this protocol:

## Step 1: Analyze and Decompose

Analyze the objective and the codebase. Decompose it into parallel tasks that can be worked on by independent agents. Consider:
- Which tasks can run in parallel vs which have dependencies
- What shared interfaces/contracts need to be agreed on upfront
- What files each task will likely touch (to avoid conflicts)

## Step 2: Initialize Session

Use the dirigent tool to initialize a coordination session:
```
dirigent init "$ARGUMENTS"
```

## Step 3: Store the Plan

Create a plan as JSON with this structure and store it:
```json
{
  "tasks": [
    {
      "id": "unique-id",
      "title": "Short title",
      "description": "Detailed description of what to build",
      "depends_on": ["other-task-id"],
      "interfaces": ["SharedInterface"]
    }
  ],
  "shared_contracts": ["Interface definitions that multiple tasks need"],
  "execution_order": ["task-ids in topological order"]
}
```

Store it with: `dirigent plan '<json>'`

## Step 4: Define Contracts

For each shared interface, register it:
```
dirigent contracts '{"name": "InterfaceName", "owner": "task-id", "description": "what it does", "definition": "the actual interface/type definition"}'
```

## Step 5: Fan Out

For each task that has no unmet dependencies, spawn a Task sub-agent with these instructions:
- The task description
- All relevant shared contracts
- Which files it should modify (and which it must NOT touch)
- Instructions to call `dirigent update <task-id> '{"status": "in_progress"}'` when starting
- Instructions to call `dirigent update <task-id> '{"status": "completed", "result": "summary", "files_modified": [...], "interfaces": [...]}'` when done
- Instructions to call `dirigent status` to check on sibling tasks if needed

Launch independent tasks in parallel using the Task tool.

## Step 6: Collect and Merge

After all tasks complete:
1. Check `dirigent status` to verify all tasks are completed
2. Review the combined changes for consistency
3. Ensure shared contracts are satisfied
4. Create a single coherent commit or set of commits

Present a summary to the user.
