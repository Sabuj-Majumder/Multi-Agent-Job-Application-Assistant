"""Tests for the Resume Analyzer Agent.

Test cases:
1. Correctly extracts skills from a sample resume text
2. Handles malformed LLM JSON response gracefully
3. resume_analyzer_agent skips gracefully when resume_text is None
"""

from unittest.mock import MagicMock, patch

import pytest

from agents.resume_analyzer_agent import (
    clean_json_response,
    extract_profile_from_text,
    parse_llm_profile,
    resume_analyzer_agent,
)
from utils.state import AgentState, CandidateProfile


class TestCleanJsonResponse:
    """Tests for JSON response cleaning."""

    def test_clean_markdown_fences(self) -> None:
        """Should strip ```json ... ``` fences."""
        raw = '```json\n{"name": "John"}\n```'
        cleaned = clean_json_response(raw)
        assert cleaned == '{"name": "John"}'

    def test_clean_plain_fences(self) -> None:
        """Should strip ``` ... ``` fences without language tag."""
        raw = '```\n{"name": "John"}\n```'
        cleaned = clean_json_response(raw)
        assert cleaned == '{"name": "John"}'

    def test_clean_no_fences(self) -> None:
        """Should return unchanged when no fences present."""
        raw = '{"name": "John"}'
        cleaned = clean_json_response(raw)
        assert cleaned == '{"name": "John"}'


class TestParseLlmProfile:
    """Tests for LLM profile parsing."""

    def test_parse_valid_json(self) -> None:
        """Should parse valid JSON into a dictionary."""
        valid_json = '{"name": "John", "skills": ["Python", "ML"]}'
        result = parse_llm_profile(valid_json)
        assert result is not None
        assert result["name"] == "John"
        assert len(result["skills"]) == 2

    def test_parse_invalid_json(self) -> None:
        """Should return None for invalid JSON."""
        invalid = "This is not valid JSON at all"
        result = parse_llm_profile(invalid)
        assert result is None

    def test_parse_json_with_markdown_fences(self) -> None:
        """Should handle JSON wrapped in markdown fences."""
        fenced = '```json\n{"name": "Jane", "skills": ["AWS"]}\n```'
        result = parse_llm_profile(fenced)
        assert result is not None
        assert result["name"] == "Jane"


class TestExtractProfileFromText:
    """Tests for the LLM-based profile extraction."""

    def test_extracts_skills_from_resume(self) -> None:
        """Should correctly extract skills from a sample resume text."""
        mock_llm_response = MagicMock()
        mock_llm_response.content = '''{
            "name": "Alice Smith",
            "email": "alice@example.com",
            "skills": ["Python", "TensorFlow", "AWS", "Docker", "SQL"],
            "experience_years": 5,
            "job_titles": ["ML Engineer", "Data Scientist"],
            "education": ["MSc Computer Science, Stanford University"],
            "summary": "Experienced ML engineer with 5 years in production ML systems."
        }'''

        mock_llm = MagicMock()
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_llm_response

        with patch("agents.resume_analyzer_agent.get_primary_llm", return_value=mock_llm):
            with patch("agents.resume_analyzer_agent.ChatPromptTemplate") as MockTemplate:
                mock_template_instance = MagicMock()
                mock_template_instance.__or__ = MagicMock(return_value=mock_chain)
                MockTemplate.from_template.return_value = mock_template_instance

                profile = extract_profile_from_text("Sample resume text with Python and TensorFlow experience")

                assert isinstance(profile, CandidateProfile)
                assert profile.name == "Alice Smith"
                assert "Python"in profile.skills
                assert "TensorFlow"in profile.skills
                assert len(profile.skills) == 5
                assert profile.experience_years == 5

    def test_handles_malformed_llm_response(self) -> None:
        """Should return minimal profile when LLM JSON is malformed."""
        mock_llm_response = MagicMock()
        mock_llm_response.content = "This is not JSON at all, just some random text."

        mock_llm = MagicMock()
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_llm_response

        with patch("agents.resume_analyzer_agent.get_primary_llm", return_value=mock_llm):
            with patch("agents.resume_analyzer_agent.ChatPromptTemplate") as MockTemplate:
                mock_template_instance = MagicMock()
                mock_template_instance.__or__ = MagicMock(return_value=mock_chain)
                MockTemplate.from_template.return_value = mock_template_instance

                profile = extract_profile_from_text("Some resume text")

                assert isinstance(profile, CandidateProfile)
                assert profile.raw_text == "Some resume text"


class TestResumeAnalyzerAgent:
    """Tests for the resume_analyzer_agent LangGraph node function."""

    def test_skips_when_resume_text_is_none(self) -> None:
        """resume_analyzer_agent should skip gracefully when resume_text is None."""
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
            "active_agent": "resume_analyzer",
            "completed_agents": ["job_scraper"],
        }

        result = resume_analyzer_agent(state)

        assert result["candidate_profile"] is None
        assert "resume_analyzer"in result["completed_agents"]
        assert result["active_agent"] is None

    def test_skips_when_resume_text_is_empty(self) -> None:
        """resume_analyzer_agent should skip gracefully when resume_text is empty string."""
        state: AgentState = {
            "job_title": "AI Engineer",
            "location": "Remote",
            "num_results": 10,
            "resume_text": "",
            "raw_jobs": [],
            "jobs": [],
            "scrape_summary": "",
            "candidate_profile": None,
            "ranked_jobs": [],
            "tailored_bullets": None,
            "cover_letters": None,
            "error": None,
            "active_agent": "resume_analyzer",
            "completed_agents": ["job_scraper"],
        }

        result = resume_analyzer_agent(state)

        assert result["candidate_profile"] is None
        assert "resume_analyzer"in result["completed_agents"]
