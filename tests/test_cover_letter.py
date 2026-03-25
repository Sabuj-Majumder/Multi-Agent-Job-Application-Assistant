"""Tests for the Cover Letter Agent.

Test 1: agent returns state unchanged when candidate_profile is None
Test 2: agent stores the LLM string response under the correct job_id in cover_letters
Test 3: agent stores empty string for a job when LLM raises exception — continues to next job, logs error
"""

from unittest.mock import MagicMock, patch

import pytest

from agents.cover_letter_agent import cover_letter_agent
from utils.state import AgentState, CandidateProfile, Job


class TestCoverLetterSkips:
    """Test 1: agent skips when no candidate profile."""

    def test_returns_state_unchanged_when_profile_is_none(
        self, sample_job: Job, minimal_state: AgentState
    ) -> None:
        """Agent returns state unchanged when candidate_profile is None."""
        state = {
            **minimal_state,
            "ranked_jobs": [sample_job],
            "candidate_profile": None,
        }

        result = cover_letter_agent(state)

        # cover_letters should not be set
        assert "cover_letters" not in result or result.get("cover_letters") is None
        assert "cover_letter" in result["completed_agents"]


class TestCoverLetterStoresResponse:
    """Test 2: agent stores LLM string under correct job_id."""

    @patch("agents.cover_letter_agent.get_primary_llm")
    @patch("agents.cover_letter_agent.ChatPromptTemplate")
    def test_stores_llm_response_under_correct_job_id(
        self,
        MockTemplate,
        mock_get_llm,
        sample_job: Job,
        sample_candidate: CandidateProfile,
        minimal_state: AgentState,
    ) -> None:
        """Agent stores the LLM string response under the correct job_id key."""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm

        cover_text = (
            "I am excited to apply for the AI Engineer role at Acme AI Corp. "
            "My experience building production ML systems aligns with your needs. "
            "I would welcome the opportunity to discuss how my skills can contribute."
        )

        mock_response = MagicMock()
        mock_response.content = cover_text

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_response

        mock_template_instance = MagicMock()
        mock_template_instance.__or__ = MagicMock(return_value=mock_chain)
        MockTemplate.from_template.return_value = mock_template_instance

        state = {
            **minimal_state,
            "ranked_jobs": [sample_job],
            "candidate_profile": sample_candidate,
            "tailored_bullets": {sample_job.id: ["bullet 1", "bullet 2"]},
        }

        result = cover_letter_agent(state)

        assert "cover_letters" in result
        assert sample_job.id in result["cover_letters"]
        assert result["cover_letters"][sample_job.id] == cover_text
        assert "cover_letter" in result["completed_agents"]


class TestCoverLetterHandlesException:
    """Test 3: agent handles LLM exception gracefully."""

    @patch("agents.cover_letter_agent.get_primary_llm")
    @patch("agents.cover_letter_agent.ChatPromptTemplate")
    def test_stores_empty_string_on_exception_and_continues(
        self,
        MockTemplate,
        mock_get_llm,
        sample_candidate: CandidateProfile,
        minimal_state: AgentState,
    ) -> None:
        """Agent stores empty string for a job when LLM raises exception,
        continues to next job, and logs the error."""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm

        good_response = MagicMock()
        good_response.content = "A wonderful cover letter."

        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = [
            Exception("LLM service unavailable"),
            good_response,
        ]

        mock_template_instance = MagicMock()
        mock_template_instance.__or__ = MagicMock(return_value=mock_chain)
        MockTemplate.from_template.return_value = mock_template_instance

        job_fail = Job(
            id="job-fail",
            title="Engineer A",
            company="Co A",
            location="Remote",
            description="Desc",
            url="http://a.com",
            source="remoteok",
            fit_score=90,
        )
        job_ok = Job(
            id="job-ok",
            title="Engineer B",
            company="Co B",
            location="Remote",
            description="Desc",
            url="http://b.com",
            source="themuse",
            fit_score=80,
        )

        state = {
            **minimal_state,
            "ranked_jobs": [job_fail, job_ok],
            "candidate_profile": sample_candidate,
            "tailored_bullets": {},
        }

        result = cover_letter_agent(state)

        assert "cover_letters" in result
        # Failed job → empty string
        assert result["cover_letters"]["job-fail"] == ""
        # Successful job → actual text
        assert result["cover_letters"]["job-ok"] == "A wonderful cover letter."
        assert "cover_letter" in result["completed_agents"]
