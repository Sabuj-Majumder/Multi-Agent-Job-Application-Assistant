"""Tests for the Fit Scorer Agent.

Test 1: agent returns state unchanged when candidate_profile is None
Test 2: agent correctly parses LLM JSON {"score": 85, "reasoning": "Strong match"}
        and sets fit_score and fit_reasoning on the Job object
Test 3: agent assigns score=0 and reasoning="Could not score" when LLM returns
        malformed JSON — does not raise
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from agents.fit_scorer_agent import fit_scorer_agent
from utils.state import AgentState, CandidateProfile, Job


class TestFitScorerSkips:
    """Test 1: agent skips when no candidate profile."""

    def test_returns_state_unchanged_when_profile_is_none(
        self, sample_job: Job, minimal_state: AgentState
    ) -> None:
        """Agent returns state unchanged when candidate_profile is None."""
        state = {
            **minimal_state,
            "jobs": [sample_job],
            "candidate_profile": None,
        }

        result = fit_scorer_agent(state)

        assert result is not None
        # Jobs should be unchanged — no fit scores set
        assert len(result["ranked_jobs"]) == 1
        # Original job's fit_score/reasoning should remain as-is on the input Job
        assert "fit_scorer" in result["completed_agents"]


class TestFitScorerParsesLLM:
    """Test 2: agent parses valid LLM JSON."""

    @patch("agents.fit_scorer_agent.get_fast_llm")
    @patch("agents.fit_scorer_agent.ChatPromptTemplate")
    def test_parses_score_85_and_sets_on_job(
        self,
        MockTemplate,
        mock_get_llm,
        sample_job: Job,
        sample_candidate: CandidateProfile,
        minimal_state: AgentState,
    ) -> None:
        """Agent correctly parses {"score": 85, "reasoning": "Strong match"}."""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm

        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "score": 85,
            "reasoning": "Strong match",
        })

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_response

        mock_template_instance = MagicMock()
        mock_template_instance.__or__ = MagicMock(return_value=mock_chain)
        MockTemplate.from_template.return_value = mock_template_instance

        # Use a job without a pre-existing score
        job_no_score = sample_job.model_copy(update={"fit_score": None, "fit_reasoning": None})
        state = {
            **minimal_state,
            "jobs": [job_no_score],
            "candidate_profile": sample_candidate,
        }

        result = fit_scorer_agent(state)

        scored_jobs = result["jobs"]
        assert len(scored_jobs) == 1
        scored_job = scored_jobs[0]
        assert scored_job.fit_score == 85
        assert scored_job.fit_reasoning == "Strong match"
        assert len(result["ranked_jobs"]) == 1
        assert "fit_scorer" in result["completed_agents"]


class TestFitScorerMalformedJSON:
    """Test 3: agent handles malformed LLM JSON gracefully."""

    @patch("agents.fit_scorer_agent.get_fast_llm")
    @patch("agents.fit_scorer_agent.ChatPromptTemplate")
    def test_assigns_zero_score_on_malformed_json(
        self,
        MockTemplate,
        mock_get_llm,
        sample_job: Job,
        sample_candidate: CandidateProfile,
        minimal_state: AgentState,
    ) -> None:
        """Agent assigns score=0 and reasoning='Could not score' — does not raise."""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm

        mock_response = MagicMock()
        mock_response.content = "This is absolutely not valid JSON!"

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_response

        mock_template_instance = MagicMock()
        mock_template_instance.__or__ = MagicMock(return_value=mock_chain)
        MockTemplate.from_template.return_value = mock_template_instance

        job_no_score = sample_job.model_copy(update={"fit_score": None, "fit_reasoning": None})
        state = {
            **minimal_state,
            "jobs": [job_no_score],
            "candidate_profile": sample_candidate,
        }

        # Should NOT raise
        result = fit_scorer_agent(state)

        assert result is not None
        scored_jobs = result["jobs"]
        assert len(scored_jobs) == 1
        scored_job = scored_jobs[0]
        assert scored_job.fit_score == 0
        assert scored_job.fit_reasoning == "Could not score"
        assert "fit_scorer" in result["completed_agents"]
