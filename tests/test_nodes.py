"""Tests for individual node stubs."""

from __future__ import annotations

from dirigent.nodes.architect import architect_node
from dirigent.nodes.developer import developer_node
from dirigent.nodes.reviewer import reviewer_node
from dirigent.state import (
    DeveloperResult,
    DeveloperStatus,
    GraphState,
    PRStatus,
    SubTask,
)


class TestArchitectNode:
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
