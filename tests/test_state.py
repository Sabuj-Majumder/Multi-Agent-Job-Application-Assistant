"""Tests for the AgentState TypedDict and Pydantic models.

Test cases:
1. AgentState TypedDict accepts all required keys
2. Job Pydantic model validates correctly
3. Job model rejects invalid fit_score (must be 0-100)
"""

import pytest
from pydantic import ValidationError

from utils.state import AgentState, CandidateProfile, Job


class TestJob:
    """Tests for the Job Pydantic model."""

    def test_job_valid_creation(self) -> None:
        """Job model should accept all valid fields."""
        job = Job(
            id="test-uuid-123",
            title="AI Engineer",
            company="TechCorp",
            location="Remote",
            description="Build AI systems",
            url="https://example.com/job/1",
            source="adzuna",
            salary="$80,000 – $120,000",
            tags=["python", "ml", "aws"],
            fit_score=85,
            fit_reasoning="Strong match on skills",
        )
        assert job.title == "AI Engineer"
        assert job.company == "TechCorp"
        assert job.source == "adzuna"
        assert job.fit_score == 85
        assert len(job.tags) == 3

    def test_job_minimal_creation(self) -> None:
        """Job model should work with only required fields."""
        job = Job(
            id="test-uuid-456",
            title="Data Scientist",
            company="DataCo",
            location="New York",
            description="Analyze data",
            url="https://example.com/job/2",
            source="remoteok",
        )
        assert job.salary is None
        assert job.tags == []
        assert job.fit_score is None
        assert job.fit_reasoning is None

    def test_job_rejects_invalid_fit_score_too_high(self) -> None:
        """Job model should reject fit_score > 100."""
        with pytest.raises(ValidationError):
            Job(
                id="test-uuid",
                title="Test",
                company="Test",
                location="Test",
                description="Test",
                url="https://example.com",
                source="adzuna",
                fit_score=101,
            )

    def test_job_rejects_invalid_fit_score_negative(self) -> None:
        """Job model should reject fit_score < 0."""
        with pytest.raises(ValidationError):
            Job(
                id="test-uuid",
                title="Test",
                company="Test",
                location="Test",
                description="Test",
                url="https://example.com",
                source="adzuna",
                fit_score=-1,
            )

    def test_job_accepts_boundary_fit_scores(self) -> None:
        """Job model should accept fit_score of 0 and 100."""
        job_zero = Job(
            id="test-0",
            title="Test",
            company="Test",
            location="Test",
            description="Test",
            url="https://example.com",
            source="adzuna",
            fit_score=0,
        )
        assert job_zero.fit_score == 0

        job_hundred = Job(
            id="test-100",
            title="Test",
            company="Test",
            location="Test",
            description="Test",
            url="https://example.com",
            source="adzuna",
            fit_score=100,
        )
        assert job_hundred.fit_score == 100


class TestCandidateProfile:
    """Tests for the CandidateProfile Pydantic model."""

    def test_profile_full_creation(self) -> None:
        """CandidateProfile should accept all fields."""
        profile = CandidateProfile(
            name="John Doe",
            email="john@example.com",
            skills=["Python", "LangChain", "AWS"],
            experience_years=5,
            job_titles=["Backend Engineer", "ML Engineer"],
            education=["BSc Computer Science, MIT"],
            summary="Experienced ML engineer with 5 years of experience.",
            raw_text="Full resume text here...",
        )
        assert profile.name == "John Doe"
        assert len(profile.skills) == 3

    def test_profile_minimal_creation(self) -> None:
        """CandidateProfile should work with all default values."""
        profile = CandidateProfile()
        assert profile.name is None
        assert profile.skills == []
        assert profile.raw_text == ""


class TestAgentState:
    """Tests for the AgentState TypedDict."""

    def test_agent_state_accepts_all_keys(self) -> None:
        """AgentState should accept all defined keys."""
        state: AgentState = {
            "job_title": "AI Engineer",
            "location": "Remote",
            "num_results": 10,
            "resume_text": None,
            "raw_jobs": [],
            "jobs": [],
            "scrape_summary": "",
            "candidate_profile": None,
            "ranked_jobs": [],
            "tailored_bullets": None,
            "cover_letters": None,
            "error": None,
            "active_agent": None,
            "completed_agents": [],
        }
        assert state["job_title"] == "AI Engineer"
        assert state["location"] == "Remote"
        assert state["num_results"] == 10
        assert state["completed_agents"] == []

    def test_agent_state_partial_keys(self) -> None:
        """AgentState (total=False) should accept partial keys."""
        state: AgentState = {
            "job_title": "Data Scientist",
            "location": "New York",
            "num_results": 20,
        }
        assert state["job_title"] == "Data Scientist"
