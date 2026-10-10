package state_test

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/johannesmauritzson/dirigent/cmd/dirigent/state"
)

func TestNewSession(t *testing.T) {
	s := state.NewSession("build auth")
	if s.Objective != "build auth" {
		t.Fatalf("expected 'build auth', got %q", s.Objective)
	}
	if s.ID == "" {
		t.Fatal("expected non-empty ID")
	}
}

func TestSaveAndLoad(t *testing.T) {
	// Work in a temp dir with a .git marker
	tmp := t.TempDir()
	if err := os.Mkdir(filepath.Join(tmp, ".git"), 0755); err != nil {
		t.Fatal(err)
	}
	origDir, _ := os.Getwd()
	os.Chdir(tmp)
	defer os.Chdir(origDir)

	s := state.NewSession("test objective")
	s.Plan = &state.Plan{
		Tasks: []state.Task{
			{ID: "t1", Title: "Task 1", Status: state.StatusPending},
			{ID: "t2", Title: "Task 2", Status: state.StatusPending, DependsOn: []string{"t1"}},
		},
	}
	s.Contracts = []state.Contract{
		{Name: "AuthInterface", Owner: "t1", Description: "Auth service interface", Definition: "type AuthService interface { Login() }"},
	}

	if err := state.Save(s); err != nil {
		t.Fatalf("save: %v", err)
	}

	loaded, err := state.Load()
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	if loaded.ID != s.ID {
		t.Fatalf("ID mismatch: %q vs %q", loaded.ID, s.ID)
	}
	if loaded.Objective != "test objective" {
		t.Fatalf("objective mismatch: %q", loaded.Objective)
	}
	if len(loaded.Plan.Tasks) != 2 {
		t.Fatalf("expected 2 tasks, got %d", len(loaded.Plan.Tasks))
	}
	if loaded.Plan.Tasks[1].DependsOn[0] != "t1" {
		t.Fatal("expected t2 depends on t1")
	}
	if len(loaded.Contracts) != 1 {
		t.Fatalf("expected 1 contract, got %d", len(loaded.Contracts))
	}
}

func TestTaskUpdateRoundtrip(t *testing.T) {
	update := state.TaskUpdate{
		Status:        state.StatusCompleted,
		Result:        "implemented auth module",
		FilesModified: []string{"auth.go", "auth_test.go"},
		Interfaces:    []string{"AuthService"},
	}
	data, err := json.Marshal(update)
	if err != nil {
		t.Fatal(err)
	}

	var decoded state.TaskUpdate
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatal(err)
	}
	if decoded.Status != state.StatusCompleted {
		t.Fatalf("expected completed, got %q", decoded.Status)
	}
	if len(decoded.FilesModified) != 2 {
		t.Fatalf("expected 2 files, got %d", len(decoded.FilesModified))
	}
}

func TestPlanJSON(t *testing.T) {
	plan := state.Plan{
		Tasks: []state.Task{
			{ID: "auth", Title: "Auth module", Description: "OAuth + session", Status: state.StatusPending},
			{ID: "rate", Title: "Rate limiter", Description: "Token bucket", Status: state.StatusPending, DependsOn: []string{"auth"}},
		},
		SharedContracts: []string{"AuthService interface"},
		ExecutionOrder:  []string{"auth", "rate"},
	}

	data, err := json.Marshal(plan)
	if err != nil {
		t.Fatal(err)
	}

	var decoded state.Plan
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatal(err)
	}
	if len(decoded.Tasks) != 2 {
		t.Fatalf("expected 2 tasks, got %d", len(decoded.Tasks))
	}
	if decoded.ExecutionOrder[0] != "auth" {
		t.Fatal("expected auth first in execution order")
	}
}

func TestSessionFilePathUsesEnvWhenSet(t *testing.T) {
	want := filepath.Join(t.TempDir(), "session.json")
	t.Setenv(state.SessionEnv, want)

	if got := state.SessionFilePath(); got != want {
		t.Fatalf("expected %q, got %q", want, got)
	}
}
