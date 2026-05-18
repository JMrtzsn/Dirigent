package utils

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

var ignoredDirs = map[string]bool{
	"node_modules": true,
	".git":         true,
	"__pycache__":  true,
	".venv":        true,
	"dist":         true,
	"build":        true,
}

var keyFiles = []string{
	"README.md",
	"go.mod",
	"Makefile",
	"pyproject.toml",
	"package.json",
}

const maxLines = 80

// ScanFileTree walks the directory tree up to maxDepth, skipping ignored directories,
// and returns a newline-separated listing of relative file paths.
func ScanFileTree(root string, maxDepth int) (string, error) {
	var files []string
	root = filepath.Clean(root)

	err := filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return nil
		}
		rel, _ := filepath.Rel(root, path)
		if rel == "." {
			return nil
		}

		depth := len(strings.Split(rel, string(filepath.Separator)))
		if depth > maxDepth {
			if info.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}

		if info.IsDir() {
			if ignoredDirs[info.Name()] {
				return filepath.SkipDir
			}
			return nil
		}

		files = append(files, rel)
		return nil
	})
	if err != nil {
		return "", err
	}
	return strings.Join(files, "\n"), nil
}

// ReadKeyFiles reads common project files from root, truncating each to 80 lines.
// Missing files are skipped.
func ReadKeyFiles(root string) (map[string]string, error) {
	result := make(map[string]string)
	for _, name := range keyFiles {
		data, err := os.ReadFile(filepath.Join(root, name))
		if err != nil {
			continue
		}
		lines := strings.SplitAfter(string(data), "\n")
		if len(lines) > maxLines {
			lines = lines[:maxLines]
		}
		result[name] = strings.Join(lines, "")
	}
	return result, nil
}

// BuildRepoContext returns a markdown string with a file tree section and
// sections for each key file found in the repo.
func BuildRepoContext(path string) (string, error) {
	var sb strings.Builder

	tree, err := ScanFileTree(path, 4)
	if err != nil {
		return "", err
	}
	sb.WriteString("## File Tree\n\n")
	sb.WriteString(tree)
	sb.WriteString("\n")

	files, err := ReadKeyFiles(path)
	if err != nil {
		return "", err
	}
	for _, name := range keyFiles {
		content, ok := files[name]
		if !ok {
			continue
		}
		fmt.Fprintf(&sb, "\n## %s\n\n", name)
		sb.WriteString(content)
		sb.WriteString("\n")
	}

	return sb.String(), nil
}
