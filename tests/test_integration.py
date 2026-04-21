"""Integration test — runs the full graph with mocked LLM."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from dirigent.graph import build_graph
from dirigent.llm.config import Config
from dirigent.llm.provider import CompletionResult, LLMProvider
from dirigent.state import GraphState


def _make_architect_response() -> str:
    """JSON response the mocked architect LLM returns."""
    return json.dumps(
        [
            {
                "id": "pr-1",
                "title": "Add auth types",
                "description": "Define AuthToken interface",
                "branch_name": "feat/pr-1-auth-types",
            },
        ]
    )


def _make_developer_response() -> str:
    """JSON response the mocked developer LLM returns."""
    return json.dumps(
        [
            {
                "action": "create",
                "path": "src/auth/types.py",
                "content": "class AuthToken:\n    pass\n",
            },
        ]
    )


def _make_reviewer_response() -> str:
    """JSON response the mocked reviewer LLM returns."""
    return json.dumps(
        {
            "verdicts": [
                {
                    "developer_id": "dev-0",
                    "passed_tests": True,
                    "architectural_alignment": 5,
                    "diff_size_score": 5,
                    "notes": "Clean implementation",
                },
                {
                    "developer_id": "dev-1",
                    "passed_tests": True,
                    "architectural_alignment": 3,
                    "diff_size_score": 3,
                    "notes": "Okay",
                },
                {
                    "developer_id": "dev-2",
                    "passed_tests": True,
                    "architectural_alignment": 2,
                    "diff_size_score": 2,
                    "notes": "Messy",
                },
            ],
            "selected_developer_id": "dev-0",
            "recommendation": "dev-0 has the cleanest code",
        }
    )


class TestFullGraphStub:
    """Integration test: full graph run with stub nodes (no LLM)."""

    @patch("dirigent.graph.create_feature_branch", return_value="feature/test")
    def test_graph_runs_to_interrupt(self, _mock_fb: MagicMock) -> None:
        """Graph runs architect → setup → fan-out → developers → reviewer → interrupt."""
        checkpointer = MemorySaver()
        graph = build_graph(checkpointer=checkpointer)

        initial_state = GraphState(
            objective="Add authentication module",
            repo_path="/tmp/fake-repo",
        )
        config = {"configurable": {"thread_id": "test-stub"}}

        events = list(graph.stream(initial_state, config=config, stream_mode="updates"))

        node_names = [name for event in events for name in event]
        assert "architect" in node_names
        assert "setup_feature_branch" in node_names
        assert "reviewer" in node_names
        assert node_names.count("developer") == 3

        state = graph.get_state(config)
        assert state.next == ("human_review",)


class TestFullGraphLLM:
    """Integration test: full graph run with mocked LLM."""

    @patch("dirigent.graph.create_feature_branch", return_value="feature/test")
    @patch("dirigent.nodes.developer._run_tests", return_value="tests passed")
    @patch("dirigent.nodes.developer._commit_changes", return_value="+5 -1")
    @patch("dirigent.nodes.developer._apply_file_operations")
    @patch("dirigent.nodes.developer.WorktreeManager")
    @patch("dirigent.nodes.developer.build_repo_context", return_value="fake context")
    @patch("dirigent.nodes.architect.build_repo_context", return_value="fake context")
    def test_graph_runs_with_llm_to_interrupt(
        self,
        _mock_arch_ctx: MagicMock,
        _mock_dev_ctx: MagicMock,
        mock_wt_cls: MagicMock,
        _mock_apply: MagicMock,
        _mock_commit: MagicMock,
        _mock_tests: MagicMock,
        _mock_fb: MagicMock,
    ) -> None:
        """Full graph with mocked LLM runs to human interrupt."""
        mock_manager = MagicMock()
        mock_wt_cls.return_value = mock_manager
        mock_manager.create.return_value = MagicMock()

        call_count = {"n": 0}
        responses = [
            _make_architect_response(),
            _make_developer_response(),
            _make_developer_response(),
            _make_developer_response(),
            _make_reviewer_response(),
        ]

        def mock_complete(messages, *, model, temperature=0.0, max_tokens=4096):
            idx = min(call_count["n"], len(responses) - 1)
            response = responses[idx]
            call_count["n"] += 1
            return CompletionResult(content=response, model=model)

        mock_provider = MagicMock(spec=LLMProvider)
        mock_provider.complete.side_effect = mock_complete

        dirigent_config = Config(provider=mock_provider)
        checkpointer = MemorySaver()
        graph = build_graph(checkpointer=checkpointer)

        initial_state = GraphState(
            objective="Add authentication module",
            repo_path="/tmp/fake-repo",
        )
        config = {
            "configurable": {
                "thread_id": "test-llm",
                "dirigent_config": dirigent_config,
            }
        }

        events = list(graph.stream(initial_state, config=config, stream_mode="updates"))

        node_names = [name for event in events for name in event]
        assert "architect" in node_names
        assert "reviewer" in node_names
        assert node_names.count("developer") == 3

        state = graph.get_state(config)
        assert state.next == ("human_review",)

        assert mock_provider.complete.call_count == 5

    @patch("dirigent.graph.create_draft_pr", return_value="https://github.com/test/pr/1")
    @patch("dirigent.graph.merge_branch")
    @patch("dirigent.graph.delete_branch")
    @patch("dirigent.graph.create_feature_branch", return_value="feature/test")
    @patch("dirigent.nodes.developer._run_tests", return_value="tests passed")
    @patch("dirigent.nodes.developer._commit_changes", return_value="+5 -1")
    @patch("dirigent.nodes.developer._apply_file_operations")
    @patch("dirigent.nodes.developer.WorktreeManager")
    @patch("dirigent.nodes.developer.build_repo_context", return_value="fake context")
    @patch("dirigent.nodes.architect.build_repo_context", return_value="fake context")
    def test_human_approval_merges_and_finalizes(
        self,
        _mock_arch_ctx: MagicMock,
        _mock_dev_ctx: MagicMock,
        mock_wt_cls: MagicMock,
        _mock_apply: MagicMock,
        _mock_commit: MagicMock,
        _mock_tests: MagicMock,
        _mock_fb: MagicMock,
        _mock_del: MagicMock,
        mock_merge: MagicMock,
        mock_pr: MagicMock,
    ) -> None:
        """After human approves, graph merges winner and creates draft PR."""
        mock_manager = MagicMock()
        mock_wt_cls.return_value = mock_manager
        mock_manager.create.return_value = MagicMock()

        call_count = {"n": 0}
        responses = [
            _make_architect_response(),
            _make_developer_response(),
            _make_developer_response(),
            _make_developer_response(),
            _make_reviewer_response(),
        ]

        def mock_complete(messages, *, model, temperature=0.0, max_tokens=4096):
            idx = min(call_count["n"], len(responses) - 1)
            response = responses[idx]
            call_count["n"] += 1
            return CompletionResult(content=response, model=model)

        mock_provider = MagicMock(spec=LLMProvider)
        mock_provider.complete.side_effect = mock_complete

        dirigent_config = Config(provider=mock_provider)
        checkpointer = MemorySaver()
        graph = build_graph(checkpointer=checkpointer)

        initial_state = GraphState(
            objective="Add auth",
            repo_path="/tmp/fake-repo",
        )
        config = {
            "configurable": {
                "thread_id": "test-approval",
                "dirigent_config": dirigent_config,
            }
        }

        # Run to interrupt
        list(graph.stream(initial_state, config=config, stream_mode="updates"))
        state = graph.get_state(config)
        assert state.next == ("human_review",)

        # Resume with approval
        events = list(
            graph.stream(
                Command(resume="yes"),
                config=config,
                stream_mode="updates",
            )
        )

        node_names = [name for event in events for name in event]
        assert "merge_winner" in node_names
        assert "finalize" in node_names

        # Verify merge was called
        mock_merge.assert_called_once()

        # Verify draft PR was created
        mock_pr.assert_called_once()

        # Graph should be complete
        final_state = graph.get_state(config)
        assert final_state.next == ()
