"""Tests for repository scanning utilities."""

from __future__ import annotations

from pathlib import Path

from dirigent.utils.repo import build_repo_context, read_key_files, scan_file_tree


class TestScanFileTree:
    def test_scans_git_repo(self, tmp_path: Path) -> None:
        """Should list files in a git repo via git ls-files fallback."""
        # Create a minimal file structure (not a git repo, so fallback walk)
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("print('hello')")
        (tmp_path / "README.md").write_text("# Test")

        tree = scan_file_tree(tmp_path)
        assert "README.md" in tree
        assert "src/" in tree or "src/main.py" in tree

    def test_respects_max_depth(self, tmp_path: Path) -> None:
        deep = tmp_path / "a" / "b" / "c" / "d" / "e"
        deep.mkdir(parents=True)
        (deep / "deep.txt").write_text("deep")
        (tmp_path / "shallow.txt").write_text("shallow")

        tree = scan_file_tree(tmp_path, max_depth=2)
        assert "shallow.txt" in tree
        # The deep file should be excluded at depth 4+
        assert "deep.txt" not in tree

    def test_skips_ignored_dirs(self, tmp_path: Path) -> None:
        (tmp_path / "node_modules").mkdir()
        (tmp_path / "node_modules" / "pkg.json").write_text("{}")
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "app.py").write_text("")

        tree = scan_file_tree(tmp_path)
        assert "node_modules" not in tree
        assert "app.py" in tree


class TestReadKeyFiles:
    def test_reads_existing_key_files(self, tmp_path: Path) -> None:
        (tmp_path / "README.md").write_text("# My Project")
        (tmp_path / "pyproject.toml").write_text('[project]\nname = "test"')

        files = read_key_files(tmp_path)
        assert "README.md" in files
        assert "pyproject.toml" in files
        assert "# My Project" in files["README.md"]

    def test_ignores_missing_files(self, tmp_path: Path) -> None:
        files = read_key_files(tmp_path)
        assert files == {}

    def test_truncates_long_files(self, tmp_path: Path) -> None:
        long_content = "\n".join(f"line {i}" for i in range(200))
        (tmp_path / "README.md").write_text(long_content)

        files = read_key_files(tmp_path)
        lines = files["README.md"].splitlines()
        assert len(lines) == 80  # _MAX_KEY_FILE_LINES


class TestBuildRepoContext:
    def test_builds_context_string(self, tmp_path: Path) -> None:
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("print('hi')")
        (tmp_path / "README.md").write_text("# Hello")

        context = build_repo_context(str(tmp_path))
        assert "## File Tree" in context
        assert "## README.md" in context
        assert "# Hello" in context

    def test_handles_nonexistent_path(self) -> None:
        context = build_repo_context("/nonexistent/path/xyz")
        assert "does not exist" in context
