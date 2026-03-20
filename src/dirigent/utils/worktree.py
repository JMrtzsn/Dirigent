"""Git worktree management for developer isolation.

Each developer agent works in its own git worktree to prevent file conflicts
during parallel execution. This module handles the lifecycle of worktrees.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


class WorktreeError(Exception):
    """Raised when a git worktree operation fails."""


@dataclass
class Worktree:
    """Represents an active git worktree."""

    path: Path
    branch: str
    repo_path: Path

    def run_in_worktree(
        self, cmd: list[str], *, timeout: int = 120
    ) -> subprocess.CompletedProcess:
        """Execute a command inside this worktree's directory."""
        return subprocess.run(
            cmd,
            cwd=self.path,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )


class WorktreeManager:
    """Manages git worktrees for parallel developer agents.

    Usage:
        manager = WorktreeManager(Path("/path/to/repo"))
        wt = manager.create("feat/my-branch", "dev-0")
        # ... do work in wt.path ...
        manager.remove(wt)
        manager.cleanup_all()
    """

    def __init__(self, repo_path: Path) -> None:
        self.repo_path = repo_path
        self._worktrees: dict[str, Worktree] = {}

    def _git(self, *args: str) -> subprocess.CompletedProcess:
        """Run a git command in the main repo."""
        cmd = ["git", "-C", str(self.repo_path), *args]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
        if result.returncode != 0:
            logger.error("git %s failed: %s", " ".join(args), result.stderr.strip())
        return result

    def create(self, branch: str, developer_id: str) -> Worktree:
        """Create a new worktree for a developer on the given branch.

        The worktree is placed in a `.worktrees/` directory alongside the repo.
        """
        worktree_dir = self.repo_path.parent / ".worktrees" / developer_id
        worktree_dir.parent.mkdir(parents=True, exist_ok=True)

        # Remove stale worktree if it exists
        if worktree_dir.exists():
            self._git("worktree", "remove", str(worktree_dir), "--force")

        # Create new branch and worktree
        result = self._git("worktree", "add", "-b", branch, str(worktree_dir))
        if result.returncode != 0:
            # Branch might already exist — try without -b
            result = self._git("worktree", "add", str(worktree_dir), branch)
            if result.returncode != 0:
                raise WorktreeError(
                    f"Failed to create worktree for {developer_id}: {result.stderr}"
                )

        wt = Worktree(path=worktree_dir, branch=branch, repo_path=self.repo_path)
        self._worktrees[developer_id] = wt
        logger.info(
            "Created worktree for %s at %s (branch: %s)", developer_id, worktree_dir, branch
        )
        return wt

    def remove(self, worktree: Worktree) -> None:
        """Remove a worktree and clean up its branch."""
        self._git("worktree", "remove", str(worktree.path), "--force")
        # Remove from tracking
        self._worktrees = {k: v for k, v in self._worktrees.items() if v.path != worktree.path}
        logger.info("Removed worktree at %s", worktree.path)

    def cleanup_all(self) -> None:
        """Remove all managed worktrees. Call this on shutdown."""
        for developer_id, wt in list(self._worktrees.items()):
            try:
                self.remove(wt)
            except Exception:
                logger.warning("Failed to clean up worktree for %s", developer_id, exc_info=True)
        self._git("worktree", "prune")

    def list_active(self) -> list[Worktree]:
        """Return all currently active worktrees."""
        return list(self._worktrees.values())
