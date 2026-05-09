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

const sessionFile = ".dirigent-session.json"

// TaskStatus represents the lifecycle of a task.
type TaskStatus string

const (
	StatusPending    TaskStatus = "pending"
	StatusInProgress TaskStatus = "in_progress"
	StatusCompleted  TaskStatus = "completed"
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
	ID        string     `json:"id"`
	Objective string     `json:"objective"`
	CreatedAt time.Time  `json:"created_at"`
	Plan      *Plan      `json:"plan,omitempty"`
	Contracts []Contract `json:"contracts,omitempty"`
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

func sessionPath() string {
	// Look for .dirigent-session.json in current directory
	// or walk up to find a git root
	dir, err := os.Getwd()
	if err != nil {
		return sessionFile
	}

	for {
		candidate := filepath.Join(dir, sessionFile)
		if _, err := os.Stat(candidate); err == nil {
			return candidate
		}
		// Check if this is a git root
		if _, err := os.Stat(filepath.Join(dir, ".git")); err == nil {
			return filepath.Join(dir, sessionFile)
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
		dir = parent
	}

	return sessionFile
}
