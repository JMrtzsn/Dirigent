package main

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"strings"

	"github.com/johannesmauritzson/dirigent/cmd/dirigent/state"
	"github.com/spf13/cobra"
)

var version = "dirigent v0.1.0"

func main() {
	var sessionFile string

	rootCmd := &cobra.Command{
		Use:   "dirigent",
		Short: "Dirigent — parallel agent coordination for OpenCode",
		Long:  "Dirigent — parallel agent coordination for OpenCode\n\nAll output is JSON for machine consumption.",
		Version: version,
		SilenceUsage:  true,
		SilenceErrors: true,
		PersistentPreRun: func(cmd *cobra.Command, args []string) {
			if sessionFile != "" {
				state.SessionFileOverride = sessionFile
			}
		},
	}

	rootCmd.PersistentFlags().StringVar(&sessionFile, "session-file", "", "path to session file (default: auto-detect)")

	rootCmd.AddCommand(initCmd())
	rootCmd.AddCommand(planCmd())
	rootCmd.AddCommand(statusCmd())
	rootCmd.AddCommand(updateCmd())
	rootCmd.AddCommand(contractsCmd())
	rootCmd.AddCommand(readyCmd())
	rootCmd.AddCommand(cleanupCmd())
	rootCmd.AddCommand(graphCmd())
	rootCmd.AddCommand(checkoutCmd())
	rootCmd.AddCommand(mergeCmd())
	rootCmd.AddCommand(listWorktreesCmd())
	rootCmd.AddCommand(verifyCmd())
	rootCmd.AddCommand(reviewCmd())
	rootCmd.AddCommand(diffCmd())

	if err := rootCmd.Execute(); err != nil {
		errJSON, _ := json.Marshal(map[string]string{"error": err.Error()})
		fmt.Fprintln(os.Stderr, string(errJSON))
		os.Exit(1)
	}
}

func initCmd() *cobra.Command {
	var requireVerification bool
	cmd := &cobra.Command{
		Use:   "init <objective>",
		Short: "Initialize a new coordination session",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			session := state.NewSession(args[0])
			session.RequireVerification = requireVerification
			if err := state.Save(session); err != nil {
				return fmt.Errorf("saving session: %w", err)
			}
			return jsonOut(session)
		},
	}
	cmd.Flags().BoolVar(&requireVerification, "require-verification", false, "require human verification before merge")
	return cmd
}

func planCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "plan '<json>'",
		Short: "Store a decomposed task plan",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			session, err := state.Load()
			if err != nil {
				return fmt.Errorf("loading session: %w", err)
			}

			var plan state.Plan
			if err := json.Unmarshal([]byte(args[0]), &plan); err != nil {
				return fmt.Errorf("parsing plan JSON: %w", err)
			}

			session.Plan = &plan
			for i := range session.Plan.Tasks {
				if session.Plan.Tasks[i].Status == "" {
					session.Plan.Tasks[i].Status = state.StatusPending
				}
			}

			if err := state.Save(session); err != nil {
				return fmt.Errorf("saving session: %w", err)
			}
			return jsonOut(session)
		},
	}
}

func statusCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "status [task-id]",
		Short: "Query task or session status",
		Args:  cobra.MaximumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
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
		},
	}
}

func updateCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "update <task-id> '<json>'",
		Short: "Update a task's status and artifacts",
		Args:  cobra.ExactArgs(2),
		RunE: func(cmd *cobra.Command, args []string) error {
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
		},
	}
}

func contractsCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "contracts [json]",
		Short: "Manage shared interface contracts",
		Args:  cobra.MaximumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			session, err := state.Load()
			if err != nil {
				return fmt.Errorf("loading session: %w", err)
			}

			if len(args) == 0 {
				return jsonOut(session.Contracts)
			}

			var contract state.Contract
			if err := json.Unmarshal([]byte(args[0]), &contract); err != nil {
				return fmt.Errorf("parsing contract JSON: %w", err)
			}

			session.Contracts = append(session.Contracts, contract)
			if err := state.Save(session); err != nil {
				return fmt.Errorf("saving session: %w", err)
			}
			return jsonOut(session.Contracts)
		},
	}
}

func readyCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "ready",
		Short: "List tasks whose dependencies are all completed",
		Args:  cobra.NoArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			session, err := state.Load()
			if err != nil {
				return fmt.Errorf("loading session: %w", err)
			}
			if session.Plan == nil {
				return fmt.Errorf("no plan exists yet")
			}

			statusMap := map[string]state.TaskStatus{}
			for _, t := range session.Plan.Tasks {
				statusMap[t.ID] = t.Status
			}

			var ready []state.Task
			for _, t := range session.Plan.Tasks {
				if t.Status != state.StatusPending {
					continue
				}
				allMet := true
				for _, dep := range t.DependsOn {
					if statusMap[dep] != state.StatusCompleted {
						allMet = false
						break
					}
				}
				if allMet {
					ready = append(ready, t)
				}
			}

			return jsonOut(ready)
		},
	}
}

func cleanupCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "cleanup",
		Short: "Remove worktrees, task branches, and the session file",
		Args:  cobra.NoArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			worktreesRemoved := 0

			// Load session to find branches/worktrees to clean
			session, err := state.Load()
			if err == nil && session.Plan != nil {
				repoRoot, _ := gitRun("rev-parse", "--show-toplevel")
				repoRoot = strings.TrimSpace(repoRoot)

				for _, t := range session.Plan.Tasks {
					if t.Branch == "" {
						continue
					}
					worktreePath := fmt.Sprintf("%s/.dirigent-worktrees/%s", repoRoot, t.ID)
					gitRun("worktree", "remove", worktreePath, "--force")
					gitRun("branch", "-D", t.Branch)
					worktreesRemoved++
				}
			}

			// Prune worktrees and remove directory
			gitRun("worktree", "prune")

			repoRoot, _ := gitRun("rev-parse", "--show-toplevel")
			repoRoot = strings.TrimSpace(repoRoot)
			os.RemoveAll(fmt.Sprintf("%s/.dirigent-worktrees", repoRoot))

			// Remove session file
			if err := os.Remove(state.SessionFilePath()); err != nil && !os.IsNotExist(err) {
				return fmt.Errorf("removing session: %w", err)
			}

			return jsonOut(map[string]any{
				"status":             "cleaned up",
				"worktrees_removed":  worktreesRemoved,
			})
		},
	}
}

func checkoutCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "checkout <task-id>",
		Short: "Create a git worktree for a task (true parallel isolation)",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			taskID := args[0]

			session, err := state.Load()
			if err != nil {
				return fmt.Errorf("loading session: %w", err)
			}
			if session.Plan == nil {
				return fmt.Errorf("no plan exists yet")
			}

			// Record base branch on first checkout
			if session.BaseBranch == "" {
				out, err := gitRun("rev-parse", "--abbrev-ref", "HEAD")
				if err != nil {
					return fmt.Errorf("detecting current branch: %w", err)
				}
				session.BaseBranch = strings.TrimSpace(out)
			}

			// Find the task
			var task *state.Task
			for i := range session.Plan.Tasks {
				if session.Plan.Tasks[i].ID == taskID {
					task = &session.Plan.Tasks[i]
					break
				}
			}
			if task == nil {
				return fmt.Errorf("task '%s' not found", taskID)
			}

			branchName := fmt.Sprintf("dirigent/%s", taskID)
			worktreeDir := fmt.Sprintf(".dirigent-worktrees/%s", taskID)

			// Get absolute path for worktree
			repoRoot, err := gitRun("rev-parse", "--show-toplevel")
			if err != nil {
				return fmt.Errorf("finding repo root: %w", err)
			}
			repoRoot = strings.TrimSpace(repoRoot)
			worktreePath := fmt.Sprintf("%s/%s", repoRoot, worktreeDir)

			// Create worktree with new branch
			if _, err := gitRun("worktree", "add", "-b", branchName, worktreePath, session.BaseBranch); err != nil {
				// Branch might already exist, try without -b
				if _, err2 := gitRun("worktree", "add", worktreePath, branchName); err2 != nil {
					return fmt.Errorf("creating worktree: %w", err)
				}
			}

			task.Branch = branchName
			task.Status = state.StatusInProgress

			if err := state.Save(session); err != nil {
				return fmt.Errorf("saving session: %w", err)
			}

			return jsonOut(map[string]string{
				"task":     taskID,
				"branch":  branchName,
				"worktree": worktreePath,
				"base":    session.BaseBranch,
				"status":  "checked out",
			})
		},
	}
}

func mergeCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "merge <task-id>",
		Short: "Merge a completed task branch and remove its worktree",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			taskID := args[0]

			session, err := state.Load()
			if err != nil {
				return fmt.Errorf("loading session: %w", err)
			}
			if session.Plan == nil {
				return fmt.Errorf("no plan exists yet")
			}

			var task *state.Task
			for i := range session.Plan.Tasks {
				if session.Plan.Tasks[i].ID == taskID {
					task = &session.Plan.Tasks[i]
					break
				}
			}
			if task == nil {
				return fmt.Errorf("task '%s' not found", taskID)
			}
			if task.Branch == "" {
				return fmt.Errorf("task '%s' has no branch", taskID)
			}
			if session.RequireVerification {
				if task.Status != state.StatusVerified {
					return fmt.Errorf("task '%s' requires verification before merge (status: %s) — run: dirigent verify %s", taskID, task.Status, taskID)
				}
			} else {
				if task.Status != state.StatusCompleted && task.Status != state.StatusVerified {
					return fmt.Errorf("task '%s' is not completed (status: %s)", taskID, task.Status)
				}
			}

			base := session.BaseBranch
			if base == "" {
				return fmt.Errorf("no base branch recorded in session")
			}

			// Remove worktree first (if it exists)
			repoRoot, _ := gitRun("rev-parse", "--show-toplevel")
			repoRoot = strings.TrimSpace(repoRoot)
			worktreePath := fmt.Sprintf("%s/.dirigent-worktrees/%s", repoRoot, taskID)
			gitRun("worktree", "remove", worktreePath) // ignore error if already removed

			// Merge task branch into base
			if _, err := gitRun("merge", "--no-ff", "-m", fmt.Sprintf("Merge dirigent task: %s", taskID), task.Branch); err != nil {
				return jsonOut(map[string]string{
					"task":   taskID,
					"status": "conflict",
					"error":  "merge conflict — resolve manually",
				})
			}

			// Delete the task branch
			gitRun("branch", "-d", task.Branch)

			if err := state.Save(session); err != nil {
				return fmt.Errorf("saving session: %w", err)
			}

			return jsonOut(map[string]string{
				"task":   taskID,
				"branch": task.Branch,
				"merged": base,
				"status": "merged",
			})
		},
	}
}

func gitRun(args ...string) (string, error) {
	cmd := exec.Command("git", args...)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return string(out), fmt.Errorf("git %s: %s", strings.Join(args, " "), strings.TrimSpace(string(out)))
	}
	return string(out), nil
}

func jsonOut(v any) error {
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	return enc.Encode(v)
}

func verifyCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "verify <task-id>",
		Short: "Mark a completed task as human-verified (unlocks merge)",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			taskID := args[0]

			session, err := state.Load()
			if err != nil {
				return fmt.Errorf("loading session: %w", err)
			}
			if session.Plan == nil {
				return fmt.Errorf("no plan exists yet")
			}

			var task *state.Task
			for i := range session.Plan.Tasks {
				if session.Plan.Tasks[i].ID == taskID {
					task = &session.Plan.Tasks[i]
					break
				}
			}
			if task == nil {
				return fmt.Errorf("task '%s' not found", taskID)
			}
			if task.Status != state.StatusCompleted {
				return fmt.Errorf("task '%s' is not completed (status: %s) — cannot verify", taskID, task.Status)
			}

			task.Status = state.StatusVerified
			if err := state.Save(session); err != nil {
				return fmt.Errorf("saving session: %w", err)
			}

			return jsonOut(map[string]string{
				"task":   taskID,
				"status": "verified",
			})
		},
	}
}

func graphCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "graph",
		Short: "Visualize the dependency graph",
		Args:  cobra.NoArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			session, err := state.Load()
			if err != nil {
				return fmt.Errorf("loading session: %w", err)
			}
			if session.Plan == nil {
				return fmt.Errorf("no plan exists yet")
			}

			tasks := session.Plan.Tasks
			statusMap := map[string]state.TaskStatus{}
			taskByID := map[string]state.Task{}
			for _, t := range tasks {
				statusMap[t.ID] = t.Status
				taskByID[t.ID] = t
			}

			// Group tasks into waves (topological layers)
			placed := map[string]bool{}
			var waves [][]state.Task
			for len(placed) < len(tasks) {
				var wave []state.Task
				for _, t := range tasks {
					if placed[t.ID] {
						continue
					}
					ready := true
					for _, dep := range t.DependsOn {
						if !placed[dep] {
							ready = false
							break
						}
					}
					if ready {
						wave = append(wave, t)
					}
				}
				if len(wave) == 0 {
					break // circular dep safety
				}
				for _, t := range wave {
					placed[t.ID] = true
				}
				waves = append(waves, wave)
			}

			// Render
			var sb strings.Builder
			sb.WriteString(fmt.Sprintf("Objective: %s\n\n", session.Objective))

			statusIcon := func(s state.TaskStatus) string {
				switch s {
				case state.StatusVerified:
					return "[V]"
				case state.StatusCompleted:
					return "[x]"
				case state.StatusInProgress:
					return "[~]"
				case state.StatusFailed:
					return "[!]"
				case state.StatusBlocked:
					return "[-]"
				default:
					return "[ ]"
				}
			}

			for i, wave := range waves {
				sb.WriteString(fmt.Sprintf("Wave %d", i+1))
				if i > 0 {
					// show which previous tasks this wave depends on
					var deps []string
					for _, t := range wave {
						deps = append(deps, t.DependsOn...)
					}
					if len(deps) > 0 {
						sb.WriteString(fmt.Sprintf(" (after: %s)", strings.Join(unique(deps), ", ")))
					}
				}
				sb.WriteString("\n")
				for _, t := range wave {
					sb.WriteString(fmt.Sprintf("  %s %s — %s\n", statusIcon(t.Status), t.ID, t.Title))
				}
				if i < len(waves)-1 {
					sb.WriteString("  |\n  v\n")
				}
			}

			fmt.Print(sb.String())
			return nil
		},
	}
}

func listWorktreesCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "list-worktrees",
		Short: "List worktrees for tasks with branches",
		Args:  cobra.NoArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			session, err := state.Load()
			if err != nil {
				return jsonOut([]any{})
			}
			if session.Plan == nil {
				return jsonOut([]any{})
			}

			repoRoot, err := gitRun("rev-parse", "--show-toplevel")
			if err != nil {
				return fmt.Errorf("finding repo root: %w", err)
			}
			repoRoot = strings.TrimSpace(repoRoot)

			type worktreeEntry struct {
				TaskID string `json:"task_id"`
				Branch string `json:"branch"`
				Path   string `json:"path"`
				Exists bool   `json:"exists"`
			}

			var entries []worktreeEntry
			for _, t := range session.Plan.Tasks {
				if t.Branch == "" {
					continue
				}
				wtPath := fmt.Sprintf("%s/.dirigent-worktrees/%s", repoRoot, t.ID)
				_, statErr := os.Stat(wtPath)
				entries = append(entries, worktreeEntry{
					TaskID: t.ID,
					Branch: t.Branch,
					Path:   wtPath,
					Exists: statErr == nil,
				})
			}

			if entries == nil {
				return jsonOut([]any{})
			}
			return jsonOut(entries)
		},
	}
}

func reviewCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "review <task-id>",
		Short: "Show task details and open worktree in a new terminal tab",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			taskID := args[0]

			session, err := state.Load()
			if err != nil {
				return fmt.Errorf("loading session: %w", err)
			}
			if session.Plan == nil {
				return fmt.Errorf("no plan exists yet")
			}

			var task *state.Task
			for i := range session.Plan.Tasks {
				if session.Plan.Tasks[i].ID == taskID {
					task = &session.Plan.Tasks[i]
					break
				}
			}
			if task == nil {
				return fmt.Errorf("task '%s' not found", taskID)
			}
			if task.Branch == "" {
				return fmt.Errorf("task '%s' has no branch/worktree", taskID)
			}

			repoRoot, _ := gitRun("rev-parse", "--show-toplevel")
			repoRoot = strings.TrimSpace(repoRoot)
			worktreePath := fmt.Sprintf("%s/.dirigent-worktrees/%s", repoRoot, taskID)

			// Get diff stat
			base := session.BaseBranch
			stat, _ := gitRun("diff", "--stat", fmt.Sprintf("%s...%s", base, task.Branch))

			fmt.Printf("Task:     %s\n", task.Title)
			fmt.Printf("Status:   %s\n", task.Status)
			fmt.Printf("Branch:   %s\n", task.Branch)
			fmt.Printf("Worktree: %s\n", worktreePath)
			fmt.Printf("\n--- Changes ---\n%s\n", strings.TrimSpace(stat))
			fmt.Printf("\nTo review:\n  cd %s\n", worktreePath)

			// Try to open worktree in Finder/Terminal (macOS)
			openCmd := exec.Command("open", worktreePath)
			if err := openCmd.Run(); err == nil {
				fmt.Printf("\n(Opened in Finder)\n")
			}

			return nil
		},
	}
}

func diffCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "diff <task-id>",
		Short: "Show git diff between base branch and task branch",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			taskID := args[0]

			session, err := state.Load()
			if err != nil {
				return fmt.Errorf("loading session: %w", err)
			}
			if session.Plan == nil {
				return fmt.Errorf("no plan exists yet")
			}

			var task *state.Task
			for i := range session.Plan.Tasks {
				if session.Plan.Tasks[i].ID == taskID {
					task = &session.Plan.Tasks[i]
					break
				}
			}
			if task == nil {
				return fmt.Errorf("task '%s' not found", taskID)
			}
			if task.Branch == "" {
				return fmt.Errorf("task has no branch")
			}

			base := session.BaseBranch
			if base == "" {
				return fmt.Errorf("no base branch recorded in session")
			}

			out, err := gitRun("diff", fmt.Sprintf("%s...%s", base, task.Branch))
			if err != nil {
				return fmt.Errorf("running git diff: %w", err)
			}
			fmt.Print(out)
			return nil
		},
	}
}

func unique(ss []string) []string {
	seen := map[string]bool{}
	var out []string
	for _, s := range ss {
		if !seen[s] {
			seen[s] = true
			out = append(out, s)
		}
	}
	return out
}
