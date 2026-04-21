"""Git operations for the SLIB (Short-Lived Integration Branch) workflow.

Handles feature branch creation, merging winner branches, and draft PR creation.
"""

from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


class GitError(Exception):
    """Raised when a git operation fails."""


def _run_git(repo_path: Path, *args: str, timeout: int = 30) -> subprocess.CompletedProcess:
    cmd = ["git", "-C", str(repo_path), *args]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    if result.returncode != 0:
        logger.error("git %s failed: %s", " ".join(args), result.stderr.strip())
    return result


def slugify(text: str) -> str:
    """Turn an objective string into a valid git branch name fragment."""
    slug = text.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    slug = slug.strip("-")
    return slug[:50]


def create_feature_branch(repo_path: Path, objective: str) -> str:
    """Create a feature branch off the current HEAD (expected to be main).

    Returns the branch name.
    """
    slug = slugify(objective)
    branch = f"feature/{slug}"

    result = _run_git(repo_path, "checkout", "-b", branch)
    if result.returncode != 0:
        raise GitError(f"Failed to create feature branch '{branch}': {result.stderr}")

    logger.info("Created feature branch: %s", branch)
    return branch


def merge_branch(repo_path: Path, source_branch: str, target_branch: str) -> None:
    """Merge source_branch into target_branch.

    Checks out the target, merges, then stays on the target.
    """
    result = _run_git(repo_path, "checkout", target_branch)
    if result.returncode != 0:
        raise GitError(f"Failed to checkout '{target_branch}': {result.stderr}")

    result = _run_git(repo_path, "merge", source_branch, "--no-edit")
    if result.returncode != 0:
        raise GitError(
            f"Failed to merge '{source_branch}' into '{target_branch}': {result.stderr}"
        )

    logger.info("Merged %s into %s", source_branch, target_branch)


def delete_branch(repo_path: Path, branch: str) -> None:
    """Delete a local branch (best-effort)."""
    _run_git(repo_path, "branch", "-D", branch)


def create_draft_pr(repo_path: Path, feature_branch: str, objective: str) -> str:
    """Push the feature branch and create a draft PR via gh CLI.

    Returns the PR URL.
    """
    # Push the feature branch
    result = _run_git(repo_path, "push", "-u", "origin", feature_branch, timeout=60)
    if result.returncode != 0:
        raise GitError(f"Failed to push '{feature_branch}': {result.stderr}")

    # Create draft PR
    cmd = [
        "gh", "pr", "create",
        "--draft",
        "--title", objective,
        "--body", f"Automated PR for: {objective}",
        "--head", feature_branch,
    ]
    result = subprocess.run(
        cmd,
        cwd=repo_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if result.returncode != 0:
        raise GitError(f"Failed to create draft PR: {result.stderr}")

    pr_url = result.stdout.strip()
    logger.info("Created draft PR: %s", pr_url)
    return pr_url
