"""Tests for individual nodes."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from dirigent.llm.config import Config
from dirigent.llm.provider import CompletionResult, LLMProvider, ProviderError
from dirigent.nodes.architect import _parse_plan, architect_node
from dirigent.nodes.developer import (
    FileOperation,
    _apply_file_operations,
    _parse_file_operations,
    developer_node,
)
from dirigent.nodes.reviewer import _parse_review, reviewer_node
from dirigent.state import (
    DeveloperResult,
    DeveloperStatus,
    GraphState,
    PRStatus,
    SubTask,
)

# --- Architect node tests ---


class TestArchitectNodeStub:
    """Tests for architect_node without LLM (stub path)."""

    def test_produces_plan_from_empty_state(self) -> None:
        state = GraphState(objective="refactor auth module")
        result = architect_node(state)
        assert "plan" in result
        assert len(result["plan"]) == 3
        assert all(isinstance(t, SubTask) for t in result["plan"])

    def test_skips_if_plan_exists(self) -> None:
        state = GraphState(
            objective="refactor auth",
            plan=[
                SubTask(
                    id="pr-1",
                    title="Existing",
                    description="Already planned",
                    branch_name="feat/existing",
                )
            ],
        )
        result = architect_node(state)
        assert result == {}


class TestArchitectNodeLLM:
    """Tests for architect_node with mocked LLM provider."""

    def _make_config(self, response_content: str) -> dict:
        """Create a RunnableConfig with a mocked provider."""
        mock_provider = MagicMock(spec=LLMProvider)
        mock_provider.complete.return_value = CompletionResult(
            content=response_content,
            model="claude-sonnet-4.6",
        )
        dirigent_config = Config(provider=mock_provider)
        return {
            "configurable": {
                "dirigent_config": dirigent_config,
            }
        }

    def test_calls_llm_and_parses_response(self) -> None:
        llm_response = json.dumps(
            [
                {
                    "id": "pr-1",
                    "title": "Add types",
                    "description": "Define auth types",
                    "branch_name": "feat/pr-1-types",
                },
                {
                    "id": "pr-2",
                    "title": "Implement logic",
                    "description": "Core auth logic",
                    "branch_name": "feat/pr-2-logic",
                },
            ]
        )
        config = self._make_config(llm_response)

        state = GraphState(objective="refactor auth", repo_path="/tmp/fake-repo")
        with patch("dirigent.nodes.architect.build_repo_context", return_value="fake context"):
            result = architect_node(state, config=config)

        assert len(result["plan"]) == 2
        assert result["plan"][0].title == "Add types"
        assert result["plan"][1].branch_name == "feat/pr-2-logic"

    def test_llm_response_with_code_fences(self) -> None:
        task = {"id": "pr-1", "title": "Fix", "description": "Fix it", "branch_name": "feat/fix"}
        llm_response = f"```json\n{json.dumps([task])}\n```"
        config = self._make_config(llm_response)

        state = GraphState(objective="fix bug", repo_path="/tmp/fake-repo")
        with patch("dirigent.nodes.architect.build_repo_context", return_value="fake"):
            result = architect_node(state, config=config)

        assert len(result["plan"]) == 1
        assert result["plan"][0].id == "pr-1"

    def test_falls_back_to_stub_on_llm_error(self) -> None:
        mock_provider = MagicMock(spec=LLMProvider)
        mock_provider.complete.side_effect = ProviderError("timeout", provider="copilot")
        dirigent_config = Config(provider=mock_provider)
        config = {"configurable": {"dirigent_config": dirigent_config}}

        state = GraphState(objective="refactor auth", repo_path="/tmp/fake-repo")
        with patch("dirigent.nodes.architect.build_repo_context", return_value="fake"):
            result = architect_node(state, config=config)

        # Should fall back to stub (3 items)
        assert len(result["plan"]) == 3

    def test_falls_back_on_invalid_json(self) -> None:
        config = self._make_config("This is not JSON at all")

        state = GraphState(objective="do stuff", repo_path="/tmp/fake")
        with patch("dirigent.nodes.architect.build_repo_context", return_value="fake"):
            result = architect_node(state, config=config)

        assert len(result["plan"]) == 3  # Stub fallback


class TestParsePlan:
    """Tests for the JSON parsing logic."""

    def test_valid_json(self) -> None:
        raw = json.dumps(
            [
                {"id": "pr-1", "title": "A", "description": "B", "branch_name": "feat/a"},
            ]
        )
        plan = _parse_plan(raw, "test")
        assert len(plan) == 1
        assert plan[0].status == PRStatus.PENDING

    def test_strips_code_fences(self) -> None:
        raw = '```json\n[{"id":"pr-1","title":"A","description":"B","branch_name":"feat/a"}]\n```'
        plan = _parse_plan(raw, "test")
        assert len(plan) == 1

    def test_skips_malformed_items(self) -> None:
        raw = json.dumps(
            [
                {"id": "pr-1", "title": "Good", "description": "OK", "branch_name": "feat/good"},
                {"id": "pr-2", "title": "Bad"},  # Missing required fields
            ]
        )
        plan = _parse_plan(raw, "test")
        assert len(plan) == 1
        assert plan[0].id == "pr-1"

    def test_empty_list_falls_back(self) -> None:
        plan = _parse_plan("[]", "test")
        assert len(plan) == 3  # Stub

    def test_non_list_falls_back(self) -> None:
        plan = _parse_plan('{"not": "a list"}', "test")
        assert len(plan) == 3  # Stub

    def test_garbage_falls_back(self) -> None:
        plan = _parse_plan("not json at all!!!", "test")
        assert len(plan) == 3  # Stub


# --- Developer node tests ---


class TestDeveloperNode:
    def test_returns_developer_result_stub(self) -> None:
        """Stub path: no config → returns stub result."""
        task = SubTask(
            id="pr-1",
            title="Test task",
            description="Do a thing",
            branch_name="feat/test",
            status=PRStatus.PENDING,
        )
        input_data = {
            "developer_id": "dev-0",
            "sub_task": task,
            "branch_name": "feat/test/dev-0",
            "repo_path": "/tmp/repo",
        }
        result = developer_node(input_data)
        assert "developer_results" in result
        assert len(result["developer_results"]) == 1
        dev_result = result["developer_results"][0]
        assert dev_result.developer_id == "dev-0"
        assert dev_result.status == DeveloperStatus.SUCCESS

    def test_stub_when_no_repo_path(self) -> None:
        """No repo_path → falls back to stub even with config."""
        mock_provider = MagicMock(spec=LLMProvider)
        dirigent_config = Config(provider=mock_provider)
        config = {"configurable": {"dirigent_config": dirigent_config}}
        task = SubTask(
            id="pr-1",
            title="Test",
            description="Desc",
            branch_name="feat/test",
        )
        input_data = {
            "developer_id": "dev-0",
            "sub_task": task,
            "branch_name": "feat/test/dev-0",
            "repo_path": "",
        }
        result = developer_node(input_data, config=config)
        assert result["developer_results"][0].status == DeveloperStatus.SUCCESS
        assert "stub" in result["developer_results"][0].diff_stats

    @patch("dirigent.nodes.developer._run_tests", return_value="tests passed")
    @patch("dirigent.nodes.developer._commit_changes", return_value="+10 -2")
    @patch("dirigent.nodes.developer._apply_file_operations")
    @patch("dirigent.nodes.developer.build_repo_context", return_value="fake context")
    @patch("dirigent.nodes.developer.WorktreeManager")
    def test_llm_path_success(
        self,
        mock_wt_cls: MagicMock,
        _mock_ctx: MagicMock,
        _mock_apply: MagicMock,
        mock_commit: MagicMock,
        mock_tests: MagicMock,
    ) -> None:
        """Full LLM path: creates worktree, calls LLM, applies ops, commits."""
        # Set up mock worktree
        mock_manager = MagicMock()
        mock_wt_cls.return_value = mock_manager
        mock_worktree = MagicMock()
        mock_manager.create.return_value = mock_worktree

        # Set up mock LLM
        llm_response = json.dumps(
            [
                {"action": "create", "path": "src/foo.py", "content": "print('hello')"},
            ]
        )
        mock_provider = MagicMock(spec=LLMProvider)
        mock_provider.complete.return_value = CompletionResult(
            content=llm_response, model="claude-haiku-4.5"
        )
        dirigent_config = Config(provider=mock_provider)
        config = {"configurable": {"dirigent_config": dirigent_config}}

        task = SubTask(
            id="pr-1",
            title="Add foo",
            description="Create foo.py",
            branch_name="feat/foo",
        )
        input_data = {
            "developer_id": "dev-0",
            "sub_task": task,
            "branch_name": "feat/foo/dev-0",
            "repo_path": "/tmp/repo",
        }

        result = developer_node(input_data, config=config)

        dev_result = result["developer_results"][0]
        assert dev_result.status == DeveloperStatus.SUCCESS
        assert dev_result.diff_stats == "+10 -2"
        assert dev_result.test_output == "tests passed"

        # Verify worktree lifecycle
        mock_manager.create.assert_called_once_with("feat/foo/dev-0", "dev-0")
        mock_manager.remove.assert_called_once_with(mock_worktree)

    @patch("dirigent.nodes.developer.build_repo_context", return_value="fake")
    @patch("dirigent.nodes.developer.WorktreeManager")
    def test_llm_path_provider_error_returns_failed(
        self, mock_wt_cls: MagicMock, _mock_ctx: MagicMock
    ) -> None:
        """LLM call fails → DeveloperResult with FAILED status."""
        mock_manager = MagicMock()
        mock_wt_cls.return_value = mock_manager
        mock_manager.create.return_value = MagicMock()

        mock_provider = MagicMock(spec=LLMProvider)
        mock_provider.complete.side_effect = ProviderError("timeout", provider="copilot")
        dirigent_config = Config(provider=mock_provider)
        config = {"configurable": {"dirigent_config": dirigent_config}}

        task = SubTask(
            id="pr-1",
            title="Fail",
            description="Will fail",
            branch_name="feat/fail",
        )
        input_data = {
            "developer_id": "dev-0",
            "sub_task": task,
            "branch_name": "feat/fail/dev-0",
            "repo_path": "/tmp/repo",
        }

        result = developer_node(input_data, config=config)
        dev_result = result["developer_results"][0]
        assert dev_result.status == DeveloperStatus.FAILED
        assert "timeout" in dev_result.error

    @patch("dirigent.nodes.developer.build_repo_context", return_value="fake")
    @patch("dirigent.nodes.developer.WorktreeManager")
    def test_worktree_cleaned_up_on_failure(
        self, mock_wt_cls: MagicMock, _mock_ctx: MagicMock
    ) -> None:
        """Worktree is removed even when LLM call fails."""
        mock_manager = MagicMock()
        mock_wt_cls.return_value = mock_manager
        mock_worktree = MagicMock()
        mock_manager.create.return_value = mock_worktree

        mock_provider = MagicMock(spec=LLMProvider)
        mock_provider.complete.side_effect = ProviderError("boom", provider="copilot")
        dirigent_config = Config(provider=mock_provider)
        config = {"configurable": {"dirigent_config": dirigent_config}}

        task = SubTask(
            id="pr-1",
            title="Boom",
            description="Explodes",
            branch_name="feat/boom",
        )
        input_data = {
            "developer_id": "dev-0",
            "sub_task": task,
            "branch_name": "feat/boom/dev-0",
            "repo_path": "/tmp/repo",
        }

        developer_node(input_data, config=config)
        mock_manager.remove.assert_called_once_with(mock_worktree)


class TestParseFileOperations:
    """Tests for _parse_file_operations."""

    def test_valid_json(self) -> None:
        raw = json.dumps(
            [
                {"action": "create", "path": "foo.py", "content": "x = 1"},
                {"action": "overwrite", "path": "bar.py", "content": "y = 2"},
            ]
        )
        ops = _parse_file_operations(raw)
        assert len(ops) == 2
        assert ops[0].action == "create"
        assert ops[0].path == "foo.py"
        assert ops[1].content == "y = 2"

    def test_strips_code_fences(self) -> None:
        raw = '```json\n[{"action":"create","path":"x.py","content":"pass"}]\n```'
        ops = _parse_file_operations(raw)
        assert len(ops) == 1

    def test_delete_operation(self) -> None:
        raw = json.dumps([{"action": "delete", "path": "old.py"}])
        ops = _parse_file_operations(raw)
        assert len(ops) == 1
        assert ops[0].action == "delete"
        assert ops[0].content == ""

    def test_skips_invalid_actions(self) -> None:
        raw = json.dumps(
            [
                {"action": "create", "path": "good.py", "content": "ok"},
                {"action": "rename", "path": "bad.py"},  # Invalid action
                {"action": "create", "path": "", "content": "no path"},  # Empty path
            ]
        )
        ops = _parse_file_operations(raw)
        assert len(ops) == 1
        assert ops[0].path == "good.py"

    def test_raises_on_garbage(self) -> None:
        try:
            _parse_file_operations("not json at all")
            msg = "Expected ProviderError"
            raise AssertionError(msg)
        except ProviderError:
            pass

    def test_raises_on_empty_list(self) -> None:
        try:
            _parse_file_operations("[]")
            msg = "Expected ProviderError"
            raise AssertionError(msg)
        except ProviderError:
            pass

    def test_raises_on_non_list(self) -> None:
        try:
            _parse_file_operations('{"not": "a list"}')
            msg = "Expected ProviderError"
            raise AssertionError(msg)
        except ProviderError:
            pass


class TestApplyFileOperations:
    """Tests for _apply_file_operations."""

    def test_create_file(self, tmp_path: MagicMock) -> None:
        worktree = MagicMock()
        worktree.path = tmp_path
        ops = [FileOperation(action="create", path="src/new.py", content="x = 1\n")]
        _apply_file_operations(worktree, ops)
        created = tmp_path / "src" / "new.py"
        assert created.exists()
        assert created.read_text() == "x = 1\n"

    def test_overwrite_file(self, tmp_path: MagicMock) -> None:
        worktree = MagicMock()
        worktree.path = tmp_path
        target = tmp_path / "existing.py"
        target.write_text("old content")
        ops = [FileOperation(action="overwrite", path="existing.py", content="new")]
        _apply_file_operations(worktree, ops)
        assert target.read_text() == "new"

    def test_delete_file(self, tmp_path: MagicMock) -> None:
        worktree = MagicMock()
        worktree.path = tmp_path
        target = tmp_path / "doomed.py"
        target.write_text("bye")
        ops = [FileOperation(action="delete", path="doomed.py")]
        _apply_file_operations(worktree, ops)
        assert not target.exists()

    def test_delete_nonexistent_is_noop(self, tmp_path: MagicMock) -> None:
        worktree = MagicMock()
        worktree.path = tmp_path
        ops = [FileOperation(action="delete", path="ghost.py")]
        _apply_file_operations(worktree, ops)  # Should not raise


# --- Reviewer node tests ---


class TestReviewerNodeStub:
    """Tests for reviewer_node without LLM (heuristic path)."""

    def test_selects_best_developer(self) -> None:
        state = GraphState(
            developer_results=[
                DeveloperResult(
                    developer_id="dev-0",
                    sub_task_id="pr-1",
                    branch_name="b0",
                    status=DeveloperStatus.SUCCESS,
                ),
                DeveloperResult(
                    developer_id="dev-1",
                    sub_task_id="pr-1",
                    branch_name="b1",
                    status=DeveloperStatus.SUCCESS,
                ),
            ]
        )
        result = reviewer_node(state)
        assert "review" in result
        review = result["review"]
        assert review.selected_developer_id != ""
        assert len(review.verdicts) == 2
        assert not review.should_retry

    def test_requests_retry_on_all_failures(self) -> None:
        state = GraphState(
            developer_results=[
                DeveloperResult(
                    developer_id="dev-0",
                    sub_task_id="pr-1",
                    branch_name="b0",
                    status=DeveloperStatus.FAILED,
                ),
            ]
        )
        result = reviewer_node(state)
        assert result["review"].should_retry is True


class TestReviewerNodeLLM:
    """Tests for reviewer_node with mocked LLM provider."""

    def _make_config(self, response_content: str) -> dict:
        """Create a RunnableConfig with a mocked provider."""
        mock_provider = MagicMock(spec=LLMProvider)
        mock_provider.complete.return_value = CompletionResult(
            content=response_content,
            model="claude-sonnet-4.6",
        )
        dirigent_config = Config(provider=mock_provider)
        return {"configurable": {"dirigent_config": dirigent_config}}

    def test_calls_llm_and_parses_verdicts(self) -> None:
        llm_response = json.dumps(
            {
                "verdicts": [
                    {
                        "developer_id": "dev-0",
                        "passed_tests": True,
                        "architectural_alignment": 5,
                        "diff_size_score": 4,
                        "notes": "Excellent implementation",
                    },
                    {
                        "developer_id": "dev-1",
                        "passed_tests": True,
                        "architectural_alignment": 3,
                        "diff_size_score": 2,
                        "notes": "Too many changes",
                    },
                ],
                "selected_developer_id": "dev-0",
                "recommendation": "dev-0 is better",
            }
        )
        config = self._make_config(llm_response)

        state = GraphState(
            developer_results=[
                DeveloperResult(
                    developer_id="dev-0",
                    sub_task_id="pr-1",
                    branch_name="b0",
                    status=DeveloperStatus.SUCCESS,
                    diff_stats="+10 -2",
                    test_output="All passed",
                ),
                DeveloperResult(
                    developer_id="dev-1",
                    sub_task_id="pr-1",
                    branch_name="b1",
                    status=DeveloperStatus.SUCCESS,
                    diff_stats="+100 -50",
                    test_output="All passed",
                ),
            ]
        )
        result = reviewer_node(state, config=config)

        review = result["review"]
        assert len(review.verdicts) == 2
        assert review.selected_developer_id == "dev-0"
        assert review.verdicts[0].architectural_alignment == 5

    def test_falls_back_to_stub_on_llm_error(self) -> None:
        mock_provider = MagicMock(spec=LLMProvider)
        mock_provider.complete.side_effect = ProviderError("timeout", provider="copilot")
        dirigent_config = Config(provider=mock_provider)
        config = {"configurable": {"dirigent_config": dirigent_config}}

        state = GraphState(
            developer_results=[
                DeveloperResult(
                    developer_id="dev-0",
                    sub_task_id="pr-1",
                    branch_name="b0",
                    status=DeveloperStatus.SUCCESS,
                ),
            ]
        )
        result = reviewer_node(state, config=config)

        # Should fall back to heuristic
        assert len(result["review"].verdicts) == 1
        assert not result["review"].should_retry

    def test_falls_back_on_invalid_json(self) -> None:
        config = self._make_config("This is garbage, not JSON")
        state = GraphState(
            developer_results=[
                DeveloperResult(
                    developer_id="dev-0",
                    sub_task_id="pr-1",
                    branch_name="b0",
                    status=DeveloperStatus.SUCCESS,
                ),
            ]
        )
        result = reviewer_node(state, config=config)
        assert len(result["review"].verdicts) == 1  # Heuristic fallback


class TestParseReview:
    """Tests for the reviewer JSON parsing logic."""

    def test_valid_json(self) -> None:
        raw = json.dumps(
            {
                "verdicts": [
                    {
                        "developer_id": "dev-0",
                        "passed_tests": True,
                        "architectural_alignment": 4,
                        "diff_size_score": 5,
                        "notes": "Good",
                    }
                ],
                "selected_developer_id": "dev-0",
                "recommendation": "Best option",
            }
        )
        verdicts = _parse_review(raw)
        assert len(verdicts) == 1
        assert verdicts[0].developer_id == "dev-0"
        assert verdicts[0].architectural_alignment == 4

    def test_strips_code_fences(self) -> None:
        inner = json.dumps(
            {
                "verdicts": [
                    {
                        "developer_id": "dev-0",
                        "passed_tests": True,
                        "architectural_alignment": 3,
                        "diff_size_score": 3,
                    }
                ],
            }
        )
        raw = f"```json\n{inner}\n```"
        verdicts = _parse_review(raw)
        assert len(verdicts) == 1

    def test_missing_verdicts_key(self) -> None:
        raw = json.dumps({"no_verdicts_here": True})
        verdicts = _parse_review(raw)
        assert verdicts == []

    def test_garbage_returns_empty(self) -> None:
        verdicts = _parse_review("not json at all!!!")
        assert verdicts == []

    def test_skips_malformed_items(self) -> None:
        raw = json.dumps(
            {
                "verdicts": [
                    {
                        "developer_id": "dev-0",
                        "passed_tests": True,
                        "architectural_alignment": 4,
                        "diff_size_score": 5,
                    },
                    {"bad": "verdict"},  # Missing developer_id
                ]
            }
        )
        verdicts = _parse_review(raw)
        assert len(verdicts) == 1
        assert verdicts[0].developer_id == "dev-0"
