// Package main provides the dirigent CLI for task coordination.
//
// Subcommands:
//
//	dirigent init     — initialize a new coordination session
//	dirigent plan     — store a decomposed task plan
//	dirigent status   — query task/sibling status
//	dirigent update   — update a task's status and artifacts
//	dirigent contracts — manage shared interface contracts
package main

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/johannesmauritzson/dirigent/cmd/dirigent/state"
)

func usage() {
	fmt.Fprintln(os.Stderr, `Usage: dirigent <command> [args]

Commands:
  init        Initialize a new coordination session
  plan        Store a decomposed task plan
  status      Query task or session status
  update      Update a task's status and artifacts
  contracts   Manage shared interface contracts

All output is JSON for machine consumption.`)
	os.Exit(1)
}

func main() {
	if len(os.Args) < 2 {
		usage()
	}

	var err error
	switch os.Args[1] {
	case "init":
		err = cmdInit(os.Args[2:])
	case "plan":
		err = cmdPlan(os.Args[2:])
	case "status":
		err = cmdStatus(os.Args[2:])
	case "update":
		err = cmdUpdate(os.Args[2:])
	case "contracts":
		err = cmdContracts(os.Args[2:])
	default:
		usage()
	}

	if err != nil {
		errJSON, _ := json.Marshal(map[string]string{"error": err.Error()})
		fmt.Fprintln(os.Stderr, string(errJSON))
		os.Exit(1)
	}
}

// cmdInit creates a new session with an objective.
func cmdInit(args []string) error {
	if len(args) < 1 {
		return fmt.Errorf("usage: dirigent init <objective>")
	}
	objective := args[0]
	session := state.NewSession(objective)
	if err := state.Save(session); err != nil {
		return fmt.Errorf("saving session: %w", err)
	}
	return jsonOut(session)
}

// cmdPlan stores a task plan into the session.
func cmdPlan(args []string) error {
	if len(args) < 1 {
		return fmt.Errorf("usage: dirigent plan '<json>'")
	}

	session, err := state.Load()
	if err != nil {
		return fmt.Errorf("loading session: %w", err)
	}

	var plan state.Plan
	if err := json.Unmarshal([]byte(args[0]), &plan); err != nil {
		return fmt.Errorf("parsing plan JSON: %w", err)
	}

	session.Plan = &plan
	// Initialize task statuses
	for i := range session.Plan.Tasks {
		if session.Plan.Tasks[i].Status == "" {
			session.Plan.Tasks[i].Status = state.StatusPending
		}
	}

	if err := state.Save(session); err != nil {
		return fmt.Errorf("saving session: %w", err)
	}
	return jsonOut(session)
}

// cmdStatus returns current session state, optionally filtered by task ID.
func cmdStatus(args []string) error {
	session, err := state.Load()
	if err != nil {
		return fmt.Errorf("loading session: %w", err)
	}

	if len(args) > 0 {
		taskID := args[0]
		for _, t := range session.Plan.Tasks {
			if t.ID == taskID {
				return jsonOut(t)
			}
		}
		return fmt.Errorf("task '%s' not found", taskID)
	}

	return jsonOut(session)
}

// cmdUpdate updates a task's status and optionally its artifacts.
func cmdUpdate(args []string) error {
	if len(args) < 2 {
		return fmt.Errorf("usage: dirigent update <task-id> '<json>'")
	}
	taskID := args[0]

	session, err := state.Load()
	if err != nil {
		return fmt.Errorf("loading session: %w", err)
	}

	if session.Plan == nil {
		return fmt.Errorf("no plan exists yet")
	}

	var update state.TaskUpdate
	if err := json.Unmarshal([]byte(args[1]), &update); err != nil {
		return fmt.Errorf("parsing update JSON: %w", err)
	}

	found := false
	for i := range session.Plan.Tasks {
		if session.Plan.Tasks[i].ID == taskID {
			if update.Status != "" {
				session.Plan.Tasks[i].Status = update.Status
			}
			if update.Result != "" {
				session.Plan.Tasks[i].Result = update.Result
			}
			if len(update.FilesModified) > 0 {
				session.Plan.Tasks[i].FilesModified = update.FilesModified
			}
			if len(update.Interfaces) > 0 {
				session.Plan.Tasks[i].Interfaces = update.Interfaces
			}
			found = true
			break
		}
	}

	if !found {
		return fmt.Errorf("task '%s' not found", taskID)
	}

	if err := state.Save(session); err != nil {
		return fmt.Errorf("saving session: %w", err)
	}
	return jsonOut(session.Plan)
}

// cmdContracts lists or adds shared contracts.
func cmdContracts(args []string) error {
	session, err := state.Load()
	if err != nil {
		return fmt.Errorf("loading session: %w", err)
	}

	if len(args) == 0 {
		// List contracts
		return jsonOut(session.Contracts)
	}

	// Add a contract
	var contract state.Contract
	if err := json.Unmarshal([]byte(args[0]), &contract); err != nil {
		return fmt.Errorf("parsing contract JSON: %w", err)
	}

	session.Contracts = append(session.Contracts, contract)
	if err := state.Save(session); err != nil {
		return fmt.Errorf("saving session: %w", err)
	}
	return jsonOut(session.Contracts)
}

func jsonOut(v any) error {
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	return enc.Encode(v)
}
