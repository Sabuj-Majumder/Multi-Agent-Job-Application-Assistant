"""Tests for the Resume Tailor Agent.

Test 1: agent returns state unchanged when candidate_profile is None
Test 2: agent correctly parses LLM JSON array and stores 5 bullets under correct job_id
Test 3: agent stores empty list for a job when LLM raises exception — continues to next job
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from agents.resume_tailor_agent import resume_tailor_agent
from utils.state import AgentState, CandidateProfile, Job


class TestResumeTailorSkips:
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

        result = resume_tailor_agent(state)

        # tailored_bullets should not be set (key may or may not exist)
        assert result.get("tailored_bullets") is None
        assert "resume_tailor" in result["completed_agents"]


class TestResumeTailorParsesJSON:
    """Test 2: agent parses LLM JSON array of 5 bullets."""

    @patch("agents.resume_tailor_agent.get_primary_llm")
    @patch("agents.resume_tailor_agent.ChatPromptTemplate")
    def test_stores_five_bullets_under_correct_job_id(
        self,
        MockTemplate,
        mock_get_llm,
        sample_job: Job,
        sample_candidate: CandidateProfile,
        minimal_state: AgentState,
    ) -> None:
        """Agent correctly parses JSON array and stores 5 bullets under the correct job_id."""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm

        five_bullets = [
            "Engineered a scalable ML pipeline using Python and AWS, reducing latency by 35%.",
            "Designed RESTful APIs using FastAPI, supporting 10M daily requests.",
            "Built Docker-based CI/CD pipelines, cutting deployment time by 50%.",
            "Optimized model inference with TensorRT, achieving 2x throughput improvement.",
            "Led cross-functional team of 5 engineers to deliver NLP product on schedule.",
        ]

        mock_response = MagicMock()
        mock_response.content = json.dumps(five_bullets)

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_response

        mock_template_instance = MagicMock()
        mock_template_instance.__or__ = MagicMock(return_value=mock_chain)
        MockTemplate.from_template.return_value = mock_template_instance

        state = {
            **minimal_state,
            "ranked_jobs": [sample_job],
            "candidate_profile": sample_candidate,
        }

        result = resume_tailor_agent(state)

        assert result["tailored_bullets"] is not None
        assert sample_job.id in result["tailored_bullets"]
        assert result["tailored_bullets"][sample_job.id] == five_bullets
        assert len(result["tailored_bullets"][sample_job.id]) == 5
        assert "resume_tailor" in result["completed_agents"]


class TestResumeTailorHandlesException:
    """Test 3: agent stores empty list when LLM raises exception."""

    @patch("agents.resume_tailor_agent.get_primary_llm")
    @patch("agents.resume_tailor_agent.ChatPromptTemplate")
    def test_stores_empty_list_on_llm_exception_and_continues(
        self,
        MockTemplate,
        mock_get_llm,
        sample_candidate: CandidateProfile,
        minimal_state: AgentState,
    ) -> None:
        """Agent stores empty list for a job when LLM raises an exception,
        then continues to the next job without crashing."""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm

        # First call raises, second call succeeds
        good_response = MagicMock()
        good_response.content = json.dumps(["Bullet A", "Bullet B"])

        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = [
            Exception("LLM connection error"),
            good_response,
        ]

        mock_template_instance = MagicMock()
        mock_template_instance.__or__ = MagicMock(return_value=mock_chain)
        MockTemplate.from_template.return_value = mock_template_instance

        job_a = Job(
            id="job-fail",
            title="Engineer A",
            company="Co A",
            location="Remote",
            description="Desc A",
            url="http://a.com",
            source="remoteok",
            fit_score=90,
        )
        job_b = Job(
            id="job-ok",
            title="Engineer B",
            company="Co B",
            location="Remote",
            description="Desc B",
            url="http://b.com",
            source="themuse",
            fit_score=80,
        )

        state = {
            **minimal_state,
            "ranked_jobs": [job_a, job_b],
            "candidate_profile": sample_candidate,
        }

        result = resume_tailor_agent(state)

        assert result["tailored_bullets"] is not None
        # Failed job should have empty list
        assert result["tailored_bullets"]["job-fail"] == []
        # Successful job should have bullets
        assert result["tailored_bullets"]["job-ok"] == ["Bullet A", "Bullet B"]
        assert "resume_tailor" in result["completed_agents"]
