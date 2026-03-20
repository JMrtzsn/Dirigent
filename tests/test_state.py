"""Tests for the Dirigent graph topology."""

from __future__ import annotations

from dirigent.state import (
    DeveloperResult,
    DeveloperStatus,
    GraphState,
    PRStatus,
    SubTask,
    merge_developer_results,
)


class TestMergeDeveloperResults:
    def test_merges_new_results(self) -> None:
        existing = [
            DeveloperResult(developer_id="dev-0", sub_task_id="pr-1", branch_name="b0"),
        ]
        new = [
            DeveloperResult(developer_id="dev-1", sub_task_id="pr-1", branch_name="b1"),
        ]
        merged = merge_developer_results(existing, new)
        assert len(merged) == 2
        ids = {r.developer_id for r in merged}
        assert ids == {"dev-0", "dev-1"}

    def test_overwrites_existing_by_id(self) -> None:
        existing = [
            DeveloperResult(
                developer_id="dev-0",
                sub_task_id="pr-1",
                branch_name="b0",
                status=DeveloperStatus.RUNNING,
            ),
        ]
        new = [
            DeveloperResult(
                developer_id="dev-0",
                sub_task_id="pr-1",
                branch_name="b0",
                status=DeveloperStatus.SUCCESS,
            ),
        ]
        merged = merge_developer_results(existing, new)
        assert len(merged) == 1
        assert merged[0].status == DeveloperStatus.SUCCESS

    def test_empty_lists(self) -> None:
        assert merge_developer_results([], []) == []


class TestGraphState:
    def test_default_construction(self) -> None:
        state = GraphState()
        assert state.objective == ""
        assert state.plan == []
        assert state.current_pr_index == 0
        assert state.developer_results == []
        assert state.human_approved is False

    def test_construction_with_values(self) -> None:
        state = GraphState(
            objective="refactor auth",
            repo_path="/tmp/repo",
        )
        assert state.objective == "refactor auth"
        assert state.repo_path == "/tmp/repo"


class TestSubTask:
    def test_default_status_is_pending(self) -> None:
        task = SubTask(
            id="pr-1",
            title="Test",
            description="A test task",
            branch_name="feat/test",
        )
        assert task.status == PRStatus.PENDING
