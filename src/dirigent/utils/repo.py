"""Repository scanning utilities for providing LLM context.

Reads the file tree and key files from a repo to give the Architect
enough context to decompose tasks intelligently.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

# Files that typically contain important structural information
_KEY_FILES = [
    "README.md",
    "pyproject.toml",
    "package.json",
    "go.mod",
    "Cargo.toml",
    "Makefile",
    "Dockerfile",
    "docker-compose.yml",
]

# Directories to skip when scanning
_SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    "dist",
    "build",
    ".eggs",
    ".tox",
    ".worktrees",
}

# Max lines to read from a key file
_MAX_KEY_FILE_LINES = 80


def scan_file_tree(repo_path: Path, *, max_depth: int = 4) -> str:
    """Get the file tree of a repository using git ls-files or fallback to walk.

    Args:
        repo_path: Path to the repo root.
        max_depth: Maximum directory depth to include.

    Returns:
        A string representation of the file tree.
    """
    # Try git ls-files first (respects .gitignore)
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=repo_path,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    if result.returncode == 0 and result.stdout.strip():
        files = result.stdout.strip().splitlines()
        # Filter by depth
        files = [f for f in files if f.count("/") < max_depth]
        return "\n".join(sorted(files))

    # Fallback: manual walk
    return _walk_tree(repo_path, max_depth=max_depth)


def _walk_tree(repo_path: Path, *, max_depth: int) -> str:
    """Walk directory tree manually, skipping ignored dirs."""
    lines: list[str] = []

    def _recurse(current: Path, depth: int) -> None:
        if depth > max_depth:
            return
        try:
            entries = sorted(current.iterdir())
        except PermissionError:
            return

        for entry in entries:
            if entry.name.startswith(".") and entry.name != ".":
                continue
            if entry.is_dir():
                if entry.name in _SKIP_DIRS:
                    continue
                rel = entry.relative_to(repo_path)
                lines.append(f"{rel}/")
                _recurse(entry, depth + 1)
            else:
                rel = entry.relative_to(repo_path)
                lines.append(str(rel))

    _recurse(repo_path, 0)
    return "\n".join(lines)


def read_key_files(repo_path: Path) -> dict[str, str]:
    """Read contents of key project files that exist in the repo.

    Returns:
        Dict mapping filename to contents (truncated to _MAX_KEY_FILE_LINES).
    """
    contents: dict[str, str] = {}
    for filename in _KEY_FILES:
        filepath = repo_path / filename
        if filepath.exists() and filepath.is_file():
            try:
                lines = filepath.read_text().splitlines()[:_MAX_KEY_FILE_LINES]
                contents[filename] = "\n".join(lines)
            except OSError as exc:
                logger.warning("Failed to read %s: %s", filepath, exc)
    return contents


def build_repo_context(repo_path: str) -> str:
    """Build a complete repo context string for the LLM.

    Combines file tree and key file contents into a single prompt-ready string.

    Args:
        repo_path: Path to the repository root.

    Returns:
        Formatted string with file tree and key file contents.
    """
    path = Path(repo_path).resolve()
    if not path.is_dir():
        return f"Repository path does not exist: {repo_path}"

    sections: list[str] = []

    # File tree
    tree = scan_file_tree(path)
    sections.append(f"## File Tree\n```\n{tree}\n```")

    # Key files
    key_files = read_key_files(path)
    for filename, content in key_files.items():
        sections.append(f"## {filename}\n```\n{content}\n```")

    return "\n\n".join(sections)
