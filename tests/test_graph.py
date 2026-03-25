"""Tests for the LangGraph pipeline graph.

Test 1: compiled pipeline graph contains all 5 expected node names
Test 2: pipeline invocation with minimal valid state completes without raising
"""

from unittest.mock import MagicMock, patch

import pytest

from utils.state import AgentState


class TestGraphNodes:
    """Test 1: Graph structure validation."""

    def test_graph_contains_all_five_nodes(self) -> None:
        """Compiled pipeline should contain all 5 agent node names."""
        from graph import build_graph

        compiled = build_graph()

        # LangGraph compiled graphs expose node names
        node_names = set(compiled.nodes.keys())

        expected = {"job_scraper", "resume_analyzer", "fit_scorer", "resume_tailor", "cover_letter"}
        # __start__ and __end__ are internal LangGraph nodes
        assert expected.issubset(node_names), (
            f"Missing nodes: {expected - node_names}"
        )


class TestGraphInvocation:
    """Test 2: Minimal pipeline invocation."""

    @patch("agents.cover_letter_agent.get_primary_llm")
    @patch("agents.resume_tailor_agent.get_primary_llm")
    @patch("agents.fit_scorer_agent.get_fast_llm")
    @patch("agents.resume_analyzer_agent.get_primary_llm")
    @patch("agents.job_scraper_agent.requests.Session")
    def test_minimal_invocation_completes(
        self,
        mock_session_cls,
        mock_resume_llm,
        mock_fit_llm,
        mock_tailor_llm,
        mock_cover_llm,
        minimal_state: AgentState,
    ) -> None:
        """Pipeline invocation with no resume and no API keys should complete
        without raising and return a state dict with 'jobs' key present."""
        import requests as req_lib

        # Mock HTTP to simulate all API failure (timeout)
        mock_session = MagicMock()
        mock_session.get.side_effect = req_lib.exceptions.Timeout("timeout")
        mock_session_cls.return_value = mock_session

        from graph import build_graph

        compiled = build_graph()
        result = compiled.invoke(minimal_state)

        assert isinstance(result, dict)
        assert "jobs" in result
