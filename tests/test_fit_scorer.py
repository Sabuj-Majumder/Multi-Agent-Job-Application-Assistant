"""Unit tests for the Fit Scorer Agent.

Tests cover:
1. Agent skips scoring when candidate_profile is None
2. Agent correctly parses LLM JSON response and sets fit_score on Job
3. Agent handles malformed LLM JSON gracefully (score=0, no crash)
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from agents.fit_scorer_agent import fit_scorer_agent
from utils.state import CandidateProfile, Job


def make_test_job(**overrides) -> Job:
    """Create a test Job object with sensible defaults."""
    defaults = {
        "id": "test-job-1",
        "title": "AI Engineer",
        "company": "Acme Corp",
        "location": "Remote",
        "description": "We are looking for an AI Engineer with Python and ML experience.",
        "url": "https://example.com/job/1",
        "source": "remoteok",
        "salary": "$100,000 - $150,000",
        "tags": ["python", "ml", "ai"],
        "fit_score": None,
        "fit_reasoning": None,
    }
    defaults.update(overrides)
    return Job(**defaults)


def make_test_state(**overrides) -> dict:
    """Create a test AgentState dict with sensible defaults."""
    defaults = {
        "job_title": "AI Engineer",
        "location": "Remote",
        "num_results": 10,
        "resume_text": "Sample resume text",
        "raw_jobs": [],
        "jobs": [],
        "scrape_summary": "",
        "candidate_profile": None,
        "ranked_jobs": [],
        "tailored_bullets": None,
        "cover_letters": None,
        "error": None,
        "active_agent": "fit_scorer",
        "completed_agents": ["job_scraper", "resume_analyzer"],
    }
    defaults.update(overrides)
    return defaults


class TestFitScorerSkipsWhenNoProfile:
    """Test 1: Agent skips scoring and returns state unchanged when candidate_profile is None."""

    def test_skips_when_candidate_profile_is_none(self):
        """Agent should return state unchanged (with ranked_jobs = jobs) when profile is None."""
        jobs = [make_test_job(), make_test_job(id="test-job-2", title="ML Engineer")]
        state = make_test_state(jobs=jobs, candidate_profile=None)

        result = fit_scorer_agent(state)

        # Should not crash
        assert result is not None
        # Jobs should remain unchanged — no fit scores set
        for job in result["jobs"]:
            if isinstance(job, Job):
                assert job.fit_score is None
            else:
                assert job.get("fit_score") is None
        # ranked_jobs should be populated (copy of jobs)
        assert len(result["ranked_jobs"]) == len(jobs)
        # fit_scorer should be in completed_agents
        assert "fit_scorer" in result["completed_agents"]

    def test_skips_when_profile_is_empty(self):
        """Agent should skip scoring when profile has no skills and no job titles."""
        empty_profile = CandidateProfile(
            name="Test User",
            skills=[],
            job_titles=[],
            raw_text="some text",
        )
        jobs = [make_test_job()]
        state = make_test_state(jobs=jobs, candidate_profile=empty_profile)

        result = fit_scorer_agent(state)

        # Should not crash, jobs should not be scored
        assert result is not None
        assert "fit_scorer" in result["completed_agents"]


class TestFitScorerParsesLLMResponse:
    """Test 2: Agent correctly parses LLM JSON response and sets fit_score on Job."""

    @patch("agents.fit_scorer_agent.get_fast_llm")
    def test_parses_valid_json_and_sets_score(self, mock_get_llm):
        """Agent should parse a valid JSON response and set fit_score on Job objects."""
        # Mock LLM to return valid JSON
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "score": 85,
            "reasoning": "Strong match with Python and ML skills."
        })
        mock_llm.__or__ = MagicMock(return_value=MagicMock(invoke=MagicMock(return_value=mock_response)))
        mock_get_llm.return_value = mock_llm

        profile = CandidateProfile(
            name="Test User",
            skills=["Python", "Machine Learning", "TensorFlow"],
            experience_years=5,
            job_titles=["ML Engineer", "Data Scientist"],
            raw_text="sample resume",
        )
        jobs = [make_test_job()]
        state = make_test_state(jobs=jobs, candidate_profile=profile)

        # Patch ChatPromptTemplate to avoid actual template rendering
        with patch("agents.fit_scorer_agent.ChatPromptTemplate") as mock_prompt:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = mock_response
            mock_template_instance = MagicMock()
            mock_template_instance.__or__ = MagicMock(return_value=mock_chain)
            mock_prompt.from_template.return_value = mock_template_instance

            result = fit_scorer_agent(state)

        # Verify scores were set
        scored_jobs = result["jobs"]
        assert len(scored_jobs) == 1

        scored_job = scored_jobs[0]
        if isinstance(scored_job, Job):
            assert scored_job.fit_score == 85
            assert scored_job.fit_reasoning == "Strong match with Python and ML skills."
        else:
            assert scored_job["fit_score"] == 85
            assert scored_job["fit_reasoning"] == "Strong match with Python and ML skills."

        # ranked_jobs should be sorted descending by fit_score
        assert len(result["ranked_jobs"]) == 1
        assert "fit_scorer" in result["completed_agents"]


class TestFitScorerHandlesMalformedJSON:
    """Test 3: Agent handles malformed LLM JSON gracefully — assigns score=0, does not raise."""

    @patch("agents.fit_scorer_agent.get_fast_llm")
    def test_handles_malformed_json_gracefully(self, mock_get_llm):
        """Agent should not crash on malformed JSON — it should assign score=0."""
        # Mock LLM to return invalid JSON
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "This is not valid JSON at all!"
        mock_llm.__or__ = MagicMock(return_value=MagicMock(invoke=MagicMock(return_value=mock_response)))
        mock_get_llm.return_value = mock_llm

        profile = CandidateProfile(
            name="Test User",
            skills=["Python"],
            experience_years=3,
            job_titles=["Developer"],
            raw_text="sample resume",
        )
        jobs = [make_test_job()]
        state = make_test_state(jobs=jobs, candidate_profile=profile)

        # Patch ChatPromptTemplate
        with patch("agents.fit_scorer_agent.ChatPromptTemplate") as mock_prompt:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = mock_response
            mock_template_instance = MagicMock()
            mock_template_instance.__or__ = MagicMock(return_value=mock_chain)
            mock_prompt.from_template.return_value = mock_template_instance

            # Should NOT raise
            result = fit_scorer_agent(state)

        # Agent should not crash
        assert result is not None

        # Score should be 0 with "Could not score" reasoning
        scored_jobs = result["jobs"]
        assert len(scored_jobs) == 1

        scored_job = scored_jobs[0]
        if isinstance(scored_job, Job):
            assert scored_job.fit_score == 0
            assert scored_job.fit_reasoning == "Could not score"
        else:
            assert scored_job["fit_score"] == 0
            assert scored_job["fit_reasoning"] == "Could not score"

        # Should still be in completed_agents
        assert "fit_scorer" in result["completed_agents"]

    @patch("agents.fit_scorer_agent.get_fast_llm")
    def test_handles_json_with_markdown_fences(self, mock_get_llm):
        """Agent should strip markdown fences before parsing JSON."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = '```json\n{"score": 72, "reasoning": "Good match."}\n```'
        mock_llm.__or__ = MagicMock(return_value=MagicMock(invoke=MagicMock(return_value=mock_response)))
        mock_get_llm.return_value = mock_llm

        profile = CandidateProfile(
            name="Test User",
            skills=["Python", "AWS"],
            experience_years=4,
            job_titles=["Backend Engineer"],
            raw_text="sample resume",
        )
        jobs = [make_test_job()]
        state = make_test_state(jobs=jobs, candidate_profile=profile)

        with patch("agents.fit_scorer_agent.ChatPromptTemplate") as mock_prompt:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = mock_response
            mock_template_instance = MagicMock()
            mock_template_instance.__or__ = MagicMock(return_value=mock_chain)
            mock_prompt.from_template.return_value = mock_template_instance

            result = fit_scorer_agent(state)

        scored_job = result["jobs"][0]
        if isinstance(scored_job, Job):
            assert scored_job.fit_score == 72
            assert scored_job.fit_reasoning == "Good match."
        else:
            assert scored_job["fit_score"] == 72
