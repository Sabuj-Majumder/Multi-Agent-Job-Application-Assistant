"""Tests for the Cover Letter Agent."""

from unittest.mock import MagicMock, patch

from agents.cover_letter_agent import cover_letter_agent
from utils.state import CandidateProfile, Job

def test_cover_letter_agent_skips_when_no_profile():
    """Test 1: agent returns state unchanged when candidate_profile is None."""
    state = {
        "ranked_jobs": [
            Job(
                id="1",
                title="A",
                company="B",
                location="C",
                description="D",
                url="E",
                source="F",
            )
        ],
        "candidate_profile": None,
    }
    result = cover_letter_agent(state)
    assert "cover_letters" not in result
    assert "cover_letter" in result["completed_agents"]


@patch("agents.cover_letter_agent.get_primary_llm")
@patch("agents.cover_letter_agent.ChatPromptTemplate")
def test_cover_letter_agent_stores_response(MockTemplate, mock_get_llm):
    """Test 2: agent correctly stores the LLM string response under correct job_id key."""
    mock_llm = MagicMock()
    mock_get_llm.return_value = mock_llm
    
    mock_response = MagicMock()
    mock_response.content = "This is a cover letter."
    
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = mock_response
    
    mock_template_instance = MagicMock()
    mock_template_instance.__or__ = MagicMock(return_value=mock_chain)
    MockTemplate.from_template.return_value = mock_template_instance

    job = Job(
        id="job123",
        title="Software Engineer",
        company="Acme Corp",
        location="Remote",
        description="Write code.",
        url="http://example.com",
        source="adzuna",
    )
    profile = CandidateProfile(
        name="Test User",
        skills=["Python"],
        experience_years=5,
        job_titles=["Dev"],
    )

    state = {
        "ranked_jobs": [job],
        "candidate_profile": profile,
        "tailored_bullets": {"job123": ["Built cool things."]},
    }

    result = cover_letter_agent(state)
    assert "cover_letters" in result
    assert result["cover_letters"]["job123"] == "This is a cover letter."
    assert "cover_letter" in result["completed_agents"]


@patch("agents.cover_letter_agent.get_primary_llm")
@patch("agents.cover_letter_agent.ChatPromptTemplate")
def test_cover_letter_agent_handles_llm_exception_gracefully(MockTemplate, mock_get_llm):
    """Test 3: agent handles an LLM exception gracefully."""
    mock_llm = MagicMock()
    mock_get_llm.return_value = mock_llm
    
    mock_chain = MagicMock()
    mock_chain.invoke.side_effect = Exception("LLM connection error")
    
    mock_template_instance = MagicMock()
    mock_template_instance.__or__ = MagicMock(return_value=mock_chain)
    MockTemplate.from_template.return_value = mock_template_instance

    job = Job(
        id="job123",
        title="Software Engineer",
        company="Acme Corp",
        location="Remote",
        description="Write code.",
        url="http://example.com",
        source="adzuna",
    )
    profile = CandidateProfile(
        name="Test User",
        skills=["Python"],
        experience_years=5,
        job_titles=["Dev"],
    )

    state = {
        "ranked_jobs": [job],
        "candidate_profile": profile,
        "tailored_bullets": {},
    }

    result = cover_letter_agent(state)
    assert "cover_letters" in result
    assert result["cover_letters"]["job123"] == ""
    assert "cover_letter" in result["completed_agents"]
