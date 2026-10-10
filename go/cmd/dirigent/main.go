package main

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"syscall"
	"time"

	"github.com/johannesmauritzson/dirigent/cmd/dirigent/state"
	"github.com/spf13/cobra"
)

var version = "dirigent v0.1.0"

func main() {
	var sessionFile string

	rootCmd := &cobra.Command{
		Use:           "dirigent",
		Short:         "Dirigent — parallel agent coordination for OpenCode",
		Long:          "Dirigent — parallel agent coordination for OpenCode\n\nAll output is JSON for machine consumption.",
		Version:       version,
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
	rootCmd.AddCommand(spawnCmd())
	rootCmd.AddCommand(mergeCmd())
	rootCmd.AddCommand(listWorktreesCmd())
	rootCmd.AddCommand(verifyCmd())
	rootCmd.AddCommand(reviewCmd())
	rootCmd.AddCommand(diffCmd())
	rootCmd.AddCommand(logCmd())

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
					if update.Status == state.StatusCompleted {
						now := time.Now()
						session.Plan.Tasks[i].CompletedAt = &now
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
				"status":            "cleaned up",
				"worktrees_removed": worktreesRemoved,
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

			task := findTask(session, taskID)
			if task == nil {
				return fmt.Errorf("task '%s' not found", taskID)
			}
			worktreePath, err := checkoutTask(session, task)
			if err != nil {
				return err
			}
			if err := state.Save(session); err != nil {
				return fmt.Errorf("saving session: %w", err)
			}

			return jsonOut(map[string]string{
				"task":     taskID,
				"branch":   task.Branch,
				"worktree": worktreePath,
				"base":     session.BaseBranch,
				"status":   "checked out",
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
			if _, err := os.Stat(worktreePath); err == nil {
				if err := commitWorktree(worktreePath, taskID); err != nil {
					return err
				}
				if _, err := gitRun("worktree", "remove", worktreePath); err != nil {
					return err
				}
			}

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

func findTask(session *state.Session, id string) *state.Task {
	for i := range session.Plan.Tasks {
		if session.Plan.Tasks[i].ID == id {
			return &session.Plan.Tasks[i]
		}
	}
	return nil
}

// checkoutTask creates the task's worktree (or reuses an existing one) and
// marks the task in progress. The caller saves the session.
func checkoutTask(session *state.Session, task *state.Task) (string, error) {
	if session.BaseBranch == "" {
		out, err := gitRun("rev-parse", "--abbrev-ref", "HEAD")
		if err != nil {
			return "", fmt.Errorf("detecting current branch: %w", err)
		}
		session.BaseBranch = strings.TrimSpace(out)
	}

	repoRoot, err := gitRun("rev-parse", "--show-toplevel")
	if err != nil {
		return "", fmt.Errorf("finding repo root: %w", err)
	}
	repoRoot = strings.TrimSpace(repoRoot)
	if err := excludeDirigentFiles(repoRoot); err != nil {
		return "", err
	}

	branchName := "dirigent/" + task.ID
	worktreePath := filepath.Join(repoRoot, ".dirigent-worktrees", task.ID)
	if _, err := os.Stat(worktreePath); err != nil {
		if _, err := gitRun("worktree", "add", "-b", branchName, worktreePath, session.BaseBranch); err != nil {
			if _, err2 := gitRun("worktree", "add", worktreePath, branchName); err2 != nil {
				return "", fmt.Errorf("creating worktree: %w", err)
			}
		}
	}

	task.Branch = branchName
	task.Status = state.StatusInProgress
	return worktreePath, nil
}

// excludeDirigentFiles keeps Dirigent's worktrees and session file out of the
// host repo's status without touching its tracked .gitignore.
func excludeDirigentFiles(repoRoot string) error {
	out, err := gitRun("-C", repoRoot, "rev-parse", "--git-path", "info/exclude")
	if err != nil {
		return err
	}
	excludePath := strings.TrimSpace(out)
	if !filepath.IsAbs(excludePath) {
		excludePath = filepath.Join(repoRoot, excludePath)
	}
	data, err := os.ReadFile(excludePath)
	if err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("reading git exclude: %w", err)
	}
	existing := map[string]bool{}
	for _, line := range strings.Split(string(data), "\n") {
		existing[strings.TrimSpace(line)] = true
	}
	var add strings.Builder
	for _, entry := range []string{".dirigent-worktrees/", ".dirigent-session.json"} {
		if !existing[entry] {
			add.WriteString(entry + "\n")
		}
	}
	if add.Len() == 0 {
		return nil
	}
	prefix := ""
	if len(data) > 0 && !strings.HasSuffix(string(data), "\n") {
		prefix = "\n"
	}
	if err := os.MkdirAll(filepath.Dir(excludePath), 0755); err != nil {
		return fmt.Errorf("creating git info dir: %w", err)
	}
	if err := os.WriteFile(excludePath, append(data, prefix+add.String()...), 0644); err != nil {
		return fmt.Errorf("writing git exclude: %w", err)
	}
	return nil
}

// commitWorktree commits any uncommitted agent work so merge never drops it.
func commitWorktree(worktreePath, taskID string) error {
	out, err := gitRun("-C", worktreePath, "status", "--porcelain")
	if err != nil {
		return err
	}
	if strings.TrimSpace(out) == "" {
		return nil
	}
	if _, err := gitRun("-C", worktreePath, "add", "-A"); err != nil {
		return err
	}
	_, err = gitRun("-C", worktreePath, "commit", "-m", "dirigent task: "+taskID)
	return err
}

func spawnPrompt(session *state.Session, task *state.Task) string {
	var sb strings.Builder
	fmt.Fprintf(&sb, "You are a Dirigent worker for task %q in the objective: %s\n\n", task.ID, session.Objective)
	fmt.Fprintf(&sb, "## Task: %s\n%s\n\n", task.Title, task.Description)
	if len(task.Interfaces) > 0 {
		fmt.Fprintf(&sb, "Interfaces you own or consume: %s\n\n", strings.Join(task.Interfaces, ", "))
	}
	if len(session.Contracts) > 0 {
		contracts, _ := json.MarshalIndent(session.Contracts, "", "  ")
		fmt.Fprintf(&sb, "## Shared contracts (code against these, do not change them)\n```json\n%s\n```\n\n", contracts)
	}
	fmt.Fprintf(&sb, `## Rules
- Your working directory is an isolated git worktree. Edit files only inside it.
- Do not commit, push, or switch branches. Dirigent commits your work on merge.
- Run the project's tests/lint for what you changed before finishing.
- When done run: dirigent update %[1]s '{"status":"completed","result":"<one-line summary>","files_modified":[...]}'
- If you cannot finish run: dirigent update %[1]s '{"status":"failed","result":"<why>"}'
`, task.ID)
	return sb.String()
}

func spawnCmd() *cobra.Command {
	var model, agent string
	cmd := &cobra.Command{
		Use:   "spawn <task-id>",
		Short: "Check out a task and run a separate headless opencode in its worktree",
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
			task := findTask(session, taskID)
			if task == nil {
				return fmt.Errorf("task '%s' not found", taskID)
			}
			if task.Status != state.StatusPending && task.Status != state.StatusFailed {
				return fmt.Errorf("task '%s' is %s, only pending or failed tasks can be spawned", taskID, task.Status)
			}
			for _, dep := range task.DependsOn {
				if d := findTask(session, dep); d == nil || (d.Status != state.StatusCompleted && d.Status != state.StatusVerified) {
					return fmt.Errorf("task '%s' depends on unfinished task '%s'", taskID, dep)
				}
			}

			worktreePath, err := checkoutTask(session, task)
			if err != nil {
				return err
			}
			sessionFile, err := filepath.Abs(state.SessionFilePath())
			if err != nil {
				return fmt.Errorf("resolving session file: %w", err)
			}
			logPath := worktreePath + ".log"
			logFile, err := os.Create(logPath)
			if err != nil {
				return fmt.Errorf("creating log: %w", err)
			}
			defer logFile.Close()

			ocArgs := []string{"run", "--dir", worktreePath, "--title", "dirigent: " + taskID}
			if model != "" {
				ocArgs = append(ocArgs, "--model", model)
			}
			if agent != "" {
				ocArgs = append(ocArgs, "--agent", agent)
			}
			ocArgs = append(ocArgs, spawnPrompt(session, task))

			oc := exec.Command("opencode", ocArgs...)
			oc.Dir = worktreePath
			oc.Env = append(os.Environ(), state.SessionEnv+"="+sessionFile)
			oc.Stdout = logFile
			oc.Stderr = logFile
			oc.SysProcAttr = &syscall.SysProcAttr{Setsid: true}
			if err := oc.Start(); err != nil {
				return fmt.Errorf("starting opencode: %w", err)
			}

			task.PID = oc.Process.Pid
			task.Log = logPath
			task.Result = ""
			if err := state.Save(session); err != nil {
				return fmt.Errorf("saving session: %w", err)
			}
			return jsonOut(map[string]any{
				"task":     taskID,
				"worktree": worktreePath,
				"pid":      task.PID,
				"log":      logPath,
				"status":   "spawned",
			})
		},
	}
	cmd.Flags().StringVar(&model, "model", "", "model for the worker (provider/model)")
	cmd.Flags().StringVar(&agent, "agent", "", "agent for the worker")
	return cmd
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
	var jsonFlag bool
	var noColor bool

	cmd := &cobra.Command{
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

			// JSON output mode
			if jsonFlag {
				type jsonTask struct {
					ID        string   `json:"id"`
					Title     string   `json:"title"`
					Status    string   `json:"status"`
					DependsOn []string `json:"depends_on"`
				}
				type jsonOutput struct {
					Objective string       `json:"objective"`
					Waves     [][]jsonTask `json:"waves"`
				}
				out := jsonOutput{Objective: session.Objective}
				for _, wave := range waves {
					var jw []jsonTask
					for _, t := range wave {
						deps := t.DependsOn
						if deps == nil {
							deps = []string{}
						}
						jw = append(jw, jsonTask{
							ID:        t.ID,
							Title:     t.Title,
							Status:    string(t.Status),
							DependsOn: deps,
						})
					}
					out.Waves = append(out.Waves, jw)
				}
				return jsonOut(out)
			}

			// Render ASCII
			useColor := !noColor

			statusIcon := func(s state.TaskStatus) string {
				var icon, color string
				switch s {
				case state.StatusVerified:
					icon, color = "[V]", "\033[32m"
				case state.StatusCompleted:
					icon, color = "[x]", "\033[32m"
				case state.StatusInProgress:
					icon, color = "[~]", "\033[33m"
				case state.StatusFailed:
					icon, color = "[!]", "\033[31m"
				case state.StatusBlocked:
					icon, color = "[-]", "\033[31m"
				default:
					icon, color = "[ ]", "\033[2m"
				}
				if useColor {
					return color + icon + "\033[0m"
				}
				return icon
			}

			var sb strings.Builder
			sb.WriteString(fmt.Sprintf("Objective: %s\n\n", session.Objective))

			for i, wave := range waves {
				sb.WriteString(fmt.Sprintf("Wave %d", i+1))
				if i > 0 {
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

	cmd.Flags().BoolVar(&jsonFlag, "json", false, "output wave structure as JSON")
	cmd.Flags().BoolVar(&noColor, "no-color", false, "disable ANSI colors")

	return cmd
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

func logCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "log",
		Short: "Show completed tasks sorted by completion time",
		Args:  cobra.NoArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			session, err := state.Load()
			if err != nil {
				return fmt.Errorf("loading session: %w", err)
			}
			if session.Plan == nil {
				return fmt.Errorf("no plan exists yet")
			}

			type logEntry struct {
				ID          string    `json:"id"`
				Title       string    `json:"title"`
				Status      string    `json:"status"`
				CompletedAt time.Time `json:"completed_at"`
			}

			var entries []logEntry
			for _, t := range session.Plan.Tasks {
				if t.CompletedAt != nil {
					entries = append(entries, logEntry{
						ID:          t.ID,
						Title:       t.Title,
						Status:      string(t.Status),
						CompletedAt: *t.CompletedAt,
					})
				}
			}

			sort.Slice(entries, func(i, j int) bool {
				return entries[i].CompletedAt.Before(entries[j].CompletedAt)
			})

			if entries == nil {
				return jsonOut([]any{})
			}
			return jsonOut(entries)
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
