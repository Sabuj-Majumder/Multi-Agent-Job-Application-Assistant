"""Unit tests for the Resume Tailor Agent."""

import json
from unittest.mock import MagicMock, patch

from agents.resume_tailor_agent import resume_tailor_agent
from utils.state import AgentState, CandidateProfile, Job


def test_resume_tailor_skips_when_profile_missing():
    """Test 1: Agent returns state unchanged when candidate_profile is None."""
    state: AgentState = {
        "job_title": "Software Engineer",
        "location": "Remote",
        "num_results": 10,
        "resume_text": "Sample resume",
        "raw_jobs": [],
        "jobs": [],
        "scrape_summary": "",
        "candidate_profile": None,  # Missing profile
        "ranked_jobs": [
            Job(
                id="job123",
                title="SWE",
                company="Acme",
                location="Remote",
                description="Python developer",
                url="http://example.com/1",
                source="adzuna",
                tags=["python"],
                fit_score=95,
            )
        ],
        "tailored_bullets": None,
        "cover_letters": None,
        "error": None,
        "active_agent": "resume_tailor",
        "completed_agents": [],
    }

    result = resume_tailor_agent(state)

    assert result["tailored_bullets"] is None
    assert "resume_tailor" in result["completed_agents"]


@patch("agents.resume_tailor_agent.get_primary_llm")
def test_resume_tailor_parses_json_correctly(mock_get_llm):
    """Test 2: Agent correctly parses LLM JSON array and stores it."""
    # Mock LLM and chain
    mock_llm = MagicMock()
    mock_chain = MagicMock()
    mock_get_llm.return_value = mock_llm
    
    # We patch ChatPromptTemplate's `|` operator indirectly by mocking the invoked chain object.
    # We create a mock response for chain.invoke
    mock_response = MagicMock()
    mock_response.content = '```json\n[\n  "Bullet 1",\n  "Bullet 2"\n]\n```'
    mock_chain.invoke.return_value = mock_response

    # In resume_tailor_agent, `chain = prompt | llm`. We can mock ChatPromptTemplate.__or__
    with patch("langchain_core.prompts.ChatPromptTemplate.__or__", return_value=mock_chain):
        state: AgentState = {
            "job_title": "Software Engineer",
            "location": "Remote",
            "num_results": 10,
            "resume_text": "Sample resume text",
            "raw_jobs": [],
            "jobs": [],
            "scrape_summary": "",
            "candidate_profile": CandidateProfile(
                name="Alice", raw_text="Experienced engineer."
            ),
            "ranked_jobs": [
                Job(
                    id="job123",
                    title="SWE",
                    company="Acme",
                    location="Remote",
                    description="Python developer",
                    url="http://example.com/1",
                    source="adzuna",
                    fit_score=95,
                )
            ],
            "tailored_bullets": None,
            "cover_letters": None,
            "error": None,
            "active_agent": "resume_tailor",
            "completed_agents": [],
        }

        result = resume_tailor_agent(state)

        assert result["tailored_bullets"] is not None
        assert result["tailored_bullets"]["job123"] == ["Bullet 1", "Bullet 2"]
        assert "resume_tailor" in result["completed_agents"]


@patch("agents.resume_tailor_agent.get_primary_llm")
def test_resume_tailor_handles_malformed_json(mock_get_llm):
    """Test 3: Agent handles malformed JSON gracefully."""
    mock_llm = MagicMock()
    mock_chain = MagicMock()
    mock_get_llm.return_value = mock_llm
    
    mock_response = MagicMock()
    mock_response.content = "This is not JSON at all."
    mock_chain.invoke.return_value = mock_response

    with patch("langchain_core.prompts.ChatPromptTemplate.__or__", return_value=mock_chain):
        state: AgentState = {
            "job_title": "Software Engineer",
            "location": "Remote",
            "num_results": 10,
            "resume_text": "Sample resume text",
            "raw_jobs": [],
            "jobs": [],
            "scrape_summary": "",
            "candidate_profile": CandidateProfile(
                name="Alice", raw_text="Experienced engineer."
            ),
            "ranked_jobs": [
                Job(
                    id="job123",
                    title="SWE",
                    company="Acme",
                    location="Remote",
                    description="Python developer",
                    url="http://example.com/1",
                    source="adzuna",
                    fit_score=95,
                )
            ],
            "tailored_bullets": None,
            "cover_letters": None,
            "error": None,
            "active_agent": "resume_tailor",
            "completed_agents": [],
        }

        result = resume_tailor_agent(state)

        assert result["tailored_bullets"] is not None
        assert result["tailored_bullets"]["job123"] == []
        assert "resume_tailor" in result["completed_agents"]
