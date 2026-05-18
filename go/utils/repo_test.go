package utils

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func setupTree(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()

	dirs := []string{
		"src",
		"src/sub",
		"src/sub/deep",
		"src/sub/deep/deeper",
		"node_modules/pkg",
		".git/objects",
		"__pycache__",
	}
	for _, d := range dirs {
		os.MkdirAll(filepath.Join(dir, d), 0o755)
	}

	files := []string{
		"README.md",
		"go.mod",
		"src/main.go",
		"src/sub/lib.go",
		"src/sub/deep/x.go",
		"src/sub/deep/deeper/y.go",
		"node_modules/pkg/index.js",
		".git/objects/abc",
		"__pycache__/mod.pyc",
	}
	for _, f := range files {
		os.WriteFile(filepath.Join(dir, f), []byte("content"), 0o644)
	}
	return dir
}

func TestScanFileTreeRespectsMaxDepth(t *testing.T) {
	dir := setupTree(t)
	result, err := ScanFileTree(dir, 4)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(result, "deeper/y.go") {
		t.Error("should not include files beyond max depth")
	}
	if !strings.Contains(result, "src/sub/deep/x.go") {
		t.Error("should include files at max depth")
	}
}

func TestScanFileTreeSkipsIgnored(t *testing.T) {
	dir := setupTree(t)
	result, err := ScanFileTree(dir, 10)
	if err != nil {
		t.Fatal(err)
	}
	for _, ignored := range []string{"node_modules", ".git", "__pycache__"} {
		if strings.Contains(result, ignored) {
			t.Errorf("should skip %s", ignored)
		}
	}
}

func TestReadKeyFilesReadsAndSkips(t *testing.T) {
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "README.md"), []byte("hello"), 0o644)

	files, err := ReadKeyFiles(dir)
	if err != nil {
		t.Fatal(err)
	}
	if files["README.md"] != "hello" {
		t.Error("should read README.md")
	}
	if _, ok := files["go.mod"]; ok {
		t.Error("should skip missing files")
	}
}

func TestReadKeyFilesTruncates(t *testing.T) {
	dir := t.TempDir()
	var lines []string
	for i := 0; i < 100; i++ {
		lines = append(lines, "line")
	}
	os.WriteFile(filepath.Join(dir, "README.md"), []byte(strings.Join(lines, "\n")), 0o644)

	files, err := ReadKeyFiles(dir)
	if err != nil {
		t.Fatal(err)
	}
	count := strings.Count(files["README.md"], "\n")
	// After splitting on \n with 100 "line" entries, truncated to 80 lines
	if count > 80 {
		t.Errorf("expected at most 80 newlines, got %d", count)
	}
}

func TestBuildRepoContext(t *testing.T) {
	dir := setupTree(t)
	os.WriteFile(filepath.Join(dir, "go.mod"), []byte("module example"), 0o644)

	result, err := BuildRepoContext(dir)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(result, "## File Tree") {
		t.Error("missing File Tree section")
	}
	if !strings.Contains(result, "## README.md") {
		t.Error("missing README.md section")
	}
	if !strings.Contains(result, "## go.mod") {
		t.Error("missing go.mod section")
	}
	if strings.Contains(result, "## Makefile") {
		t.Error("should not have section for missing file")
	}
}
