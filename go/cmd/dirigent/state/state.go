// Package state manages the coordination session persisted to disk.
package state

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/google/uuid"
)

const defaultSessionFile = ".dirigent-session.json"

// SessionFileOverride allows callers to override the session file path.
// When empty, the default resolution logic is used.
var SessionFileOverride string

// TaskStatus represents the lifecycle of a task.
type TaskStatus string

const (
	StatusPending    TaskStatus = "pending"
	StatusInProgress TaskStatus = "in_progress"
	StatusCompleted  TaskStatus = "completed"
	StatusVerified   TaskStatus = "verified"
	StatusFailed     TaskStatus = "failed"
	StatusBlocked    TaskStatus = "blocked"
)

// Task is a single unit of work in the plan.
type Task struct {
	ID            string     `json:"id"`
	Title         string     `json:"title"`
	Description   string     `json:"description"`
	Status        TaskStatus `json:"status"`
	DependsOn     []string   `json:"depends_on,omitempty"`
	Interfaces    []string   `json:"interfaces,omitempty"`
	FilesModified []string   `json:"files_modified,omitempty"`
	Result        string     `json:"result,omitempty"`
	AssignedAgent string     `json:"assigned_agent,omitempty"`
	Branch        string     `json:"branch,omitempty"`
}

// Plan holds the decomposed tasks and their dependency graph.
type Plan struct {
	Tasks           []Task   `json:"tasks"`
	SharedContracts []string `json:"shared_contracts,omitempty"`
	ExecutionOrder  []string `json:"execution_order,omitempty"`
}

// Contract defines a shared interface between tasks.
type Contract struct {
	Name        string `json:"name"`
	Owner       string `json:"owner"`
	Description string `json:"description"`
	Definition  string `json:"definition"`
}

// Session is the top-level coordination state.
type Session struct {
	ID                  string     `json:"id"`
	Objective           string     `json:"objective"`
	CreatedAt           time.Time  `json:"created_at"`
	BaseBranch          string     `json:"base_branch,omitempty"`
	RequireVerification bool       `json:"require_verification,omitempty"`
	Plan                *Plan      `json:"plan,omitempty"`
	Contracts           []Contract `json:"contracts,omitempty"`
}

// TaskUpdate is the payload for updating a task.
type TaskUpdate struct {
	Status        TaskStatus `json:"status,omitempty"`
	Result        string     `json:"result,omitempty"`
	FilesModified []string   `json:"files_modified,omitempty"`
	Interfaces    []string   `json:"interfaces,omitempty"`
}

// NewSession creates a fresh session.
func NewSession(objective string) *Session {
	return &Session{
		ID:        uuid.New().String(),
		Objective: objective,
		CreatedAt: time.Now(),
	}
}

// Save persists the session to the working directory.
func Save(s *Session) error {
	data, err := json.MarshalIndent(s, "", "  ")
	if err != nil {
		return fmt.Errorf("marshalling session: %w", err)
	}
	return os.WriteFile(sessionPath(), data, 0644)
}

// Load reads the session from the working directory.
func Load() (*Session, error) {
	data, err := os.ReadFile(sessionPath())
	if err != nil {
		return nil, fmt.Errorf("reading session file: %w", err)
	}
	var s Session
	if err := json.Unmarshal(data, &s); err != nil {
		return nil, fmt.Errorf("parsing session: %w", err)
	}
	return &s, nil
}

// SessionFilePath returns the path to the session file.
func SessionFilePath() string {
	return sessionPath()
}

func sessionPath() string {
	if SessionFileOverride != "" {
		return SessionFileOverride
	}

	dir, err := os.Getwd()
	if err != nil {
		return defaultSessionFile
	}

	for {
		candidate := filepath.Join(dir, defaultSessionFile)
		if _, err := os.Stat(candidate); err == nil {
			return candidate
		}
		if _, err := os.Stat(filepath.Join(dir, ".git")); err == nil {
			return filepath.Join(dir, defaultSessionFile)
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
		dir = parent
	}

	return defaultSessionFile
}
