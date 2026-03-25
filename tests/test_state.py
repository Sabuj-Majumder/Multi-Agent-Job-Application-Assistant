"""Tests for the Pydantic data models and AgentState TypedDict.

Test 1: Job validates correctly with all required fields
Test 2: Job raises ValidationError when fit_score is outside 0–100
Test 3: AgentState TypedDict accepts a fully populated dict without error
"""

import pytest
from pydantic import ValidationError

from utils.state import AgentState, CandidateProfile, Job


class TestJobModel:
    """Tests for the Job Pydantic model."""

    def test_job_validates_with_all_fields(self, sample_job: Job) -> None:
        """Test 1: Job model accepts all valid fields and stores them correctly."""
        assert sample_job.id == "test-job-abc-123"
        assert sample_job.title == "AI Engineer"
        assert sample_job.company == "Acme AI Corp"
        assert sample_job.location == "Remote"
        assert sample_job.source == "remoteok"
        assert sample_job.salary == "$120,000 – $160,000"
        assert sample_job.fit_score == 85
        assert sample_job.fit_reasoning == "Strong match on Python and ML skills."
        assert len(sample_job.tags) == 4
        assert "python" in sample_job.tags
        assert sample_job.posted_at == "2025-12-01T10:00:00Z"

    def test_job_rejects_fit_score_outside_range(self) -> None:
        """Test 2: Job raises ValidationError for fit_score > 100 or < 0."""
        with pytest.raises(ValidationError):
            Job(
                id="bad-score-high",
                title="Test",
                company="Test",
                location="Test",
                description="Test",
                url="https://example.com",
                source="remoteok",
                fit_score=101,
            )

        with pytest.raises(ValidationError):
            Job(
                id="bad-score-low",
                title="Test",
                company="Test",
                location="Test",
                description="Test",
                url="https://example.com",
                source="remoteok",
                fit_score=-1,
            )


class TestAgentState:
    """Tests for the AgentState TypedDict."""

    def test_agent_state_accepts_full_dict(
        self, minimal_state: AgentState, sample_job: Job, sample_candidate: CandidateProfile
    ) -> None:
        """Test 3: AgentState accepts a fully populated state dict with all keys."""
        full_state: AgentState = {
            **minimal_state,
            "job_title": "AI Engineer",
            "location": "Remote",
            "num_results": 10,
            "resume_text": "Full resume text here",
            "raw_jobs": [{"title": "AI Engineer"}],
            "jobs": [sample_job],
            "scrape_summary": "Found 1 job",
            "candidate_profile": sample_candidate,
            "ranked_jobs": [sample_job],
            "tailored_bullets": {"test-job-abc-123": ["Bullet 1"]},
            "cover_letters": {"test-job-abc-123": "Dear Hiring Manager..."},
            "error": None,
            "active_agent": "job_scraper",
            "completed_agents": ["job_scraper"],
        }

        # TypedDict is a dict — verify all keys are accessible
        assert full_state["job_title"] == "AI Engineer"
        assert full_state["location"] == "Remote"
        assert full_state["num_results"] == 10
        assert full_state["resume_text"] == "Full resume text here"
        assert len(full_state["jobs"]) == 1
        assert full_state["candidate_profile"].name == "Alice Johnson"
        assert full_state["completed_agents"] == ["job_scraper"]
        assert full_state["error"] is None
