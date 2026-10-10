package main

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func run(t *testing.T, dir string, name string, args ...string) string {
	t.Helper()
	cmd := exec.Command(name, args...)
	cmd.Dir = dir
	cmd.Env = append(os.Environ(), "DIRIGENT_SESSION=")
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("%s %v: %v\n%s", name, args, err, out)
	}
	return string(out)
}

func TestMergeKeepsUncommittedWorkerChangesAndHidesDirigentFiles(t *testing.T) {
	bin := filepath.Join(t.TempDir(), "dirigent")
	run(t, ".", "go", "build", "-o", bin, ".")

	repo := t.TempDir()
	run(t, repo, "git", "init", "-q", "-b", "main")
	run(t, repo, "git", "config", "user.email", "t@example.com")
	run(t, repo, "git", "config", "user.name", "t")
	run(t, repo, "git", "commit", "-q", "--allow-empty", "-m", "init")

	run(t, repo, bin, "init", "obj")
	run(t, repo, bin, "plan", `{"tasks":[{"id":"a","title":"A"}]}`)
	run(t, repo, bin, "checkout", "a")

	worktree := filepath.Join(repo, ".dirigent-worktrees", "a")
	if err := os.WriteFile(filepath.Join(worktree, "a.txt"), []byte("hi"), 0644); err != nil {
		t.Fatal(err)
	}
	run(t, worktree, bin, "--session-file", filepath.Join(repo, ".dirigent-session.json"), "update", "a", `{"status":"completed"}`)

	if status := run(t, repo, "git", "status", "--porcelain"); strings.Contains(status, ".dirigent") {
		t.Fatalf("dirigent files visible to host repo:\n%s", status)
	}

	run(t, repo, bin, "merge", "a")

	if _, err := os.Stat(filepath.Join(repo, "a.txt")); err != nil {
		t.Fatalf("worker change missing after merge: %v", err)
	}
}
