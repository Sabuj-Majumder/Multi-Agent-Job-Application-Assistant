"""Shared pytest fixtures for the Multi-Agent Job Application Assistant test suite.

Provides reusable sample objects:
- sample_job: A fully populated Job instance
- sample_candidate: A CandidateProfile with realistic data
- minimal_state: The smallest valid AgentState with empty output defaults
"""

import pytest

from utils.state import AgentState, CandidateProfile, Job


@pytest.fixture
def sample_job() -> Job:
    """Returns a fully populated Job object with realistic test data."""
    return Job(
        id="test-job-abc-123",
        title="AI Engineer",
        company="Acme AI Corp",
        location="Remote",
        description=(
            "We are looking for an AI Engineer to design, build, and deploy "
            "machine-learning models in production. Requirements: Python, "
            "PyTorch, AWS, Docker, CI/CD pipelines."
        ),
        url="https://example.com/jobs/ai-engineer",
        source="remoteok",
        salary="$120,000 – $160,000",
        tags=["python", "pytorch", "aws", "docker"],
        fit_score=85,
        fit_reasoning="Strong match on Python and ML skills.",
        posted_at="2025-12-01T10:00:00Z",
    )


@pytest.fixture
def sample_candidate() -> CandidateProfile:
    """Returns a CandidateProfile with skills, experience_years, job_titles, raw_text."""
    return CandidateProfile(
        name="Alice Johnson",
        email="alice@example.com",
        skills=["Python", "PyTorch", "AWS", "Docker", "SQL", "LangChain"],
        experience_years=6,
        job_titles=["ML Engineer", "Data Scientist", "Backend Developer"],
        education=["MSc Computer Science, Stanford University"],
        summary="Experienced ML engineer with 6 years building production ML systems.",
        raw_text=(
            "Alice Johnson — ML Engineer with 6 years experience. "
            "Skills: Python, PyTorch, AWS, Docker, SQL, LangChain. "
            "Education: MSc Computer Science, Stanford University."
        ),
    )


@pytest.fixture
def minimal_state() -> AgentState:
    """Returns the smallest valid AgentState with empty output defaults."""
    return {
        "job_title": "AI Engineer",
        "location": "Remote",
        "num_results": 5,
        "resume_text": None,
        "date_filter": "Any time",
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
