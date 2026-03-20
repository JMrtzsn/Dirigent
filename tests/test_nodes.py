"""Tests for individual nodes."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from dirigent.llm.config import Config
from dirigent.llm.provider import CompletionResult, LLMProvider, ProviderError
from dirigent.nodes.architect import _parse_plan, architect_node
from dirigent.nodes.developer import developer_node
from dirigent.nodes.reviewer import reviewer_node
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
    def test_returns_developer_result(self) -> None:
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


# --- Reviewer node tests ---


class TestReviewerNode:
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
