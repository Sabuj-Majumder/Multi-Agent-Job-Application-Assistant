"""Tests for the Resume Analyzer Agent.

Test 1: agent correctly parses a clean LLM JSON response into CandidateProfile
Test 2: agent retries once on JSON parse failure, then falls back — LLM called exactly twice
Test 3: agent returns state unchanged when resume_text is None or empty — no LLM call
"""

from unittest.mock import MagicMock, patch, call

import pytest

from agents.resume_analyzer_agent import (
    extract_profile_from_text,
    resume_analyzer_agent,
)
from utils.state import AgentState, CandidateProfile


class TestExtractProfile:
    """Tests for LLM-based profile extraction."""

    def test_parses_clean_json_into_candidate_profile(self) -> None:
        """Test 1: clean LLM JSON → CandidateProfile with all fields populated."""
        valid_json = (
            '{"name": "Alice Smith", "email": "alice@example.com", '
            '"skills": ["Python", "TensorFlow", "AWS", "Docker", "SQL"], '
            '"experience_years": 5, '
            '"job_titles": ["ML Engineer", "Data Scientist"], '
            '"education": ["MSc Computer Science, Stanford University"], '
            '"summary": "Experienced ML engineer with 5 years in production ML."}'
        )

        mock_llm_response = MagicMock()
        mock_llm_response.content = valid_json

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_llm_response

        mock_template = MagicMock()
        mock_template.__or__ = MagicMock(return_value=mock_chain)

        with patch("agents.resume_analyzer_agent.get_primary_llm") as mock_get_llm, \
             patch("agents.resume_analyzer_agent.ChatPromptTemplate") as MockTemplate:
            mock_get_llm.return_value = MagicMock()
            MockTemplate.from_template.return_value = mock_template

            profile = extract_profile_from_text("Sample resume text")

        assert isinstance(profile, CandidateProfile)
        assert profile.name == "Alice Smith"
        assert profile.email == "alice@example.com"
        assert len(profile.skills) == 5
        assert "Python" in profile.skills
        assert "TensorFlow" in profile.skills
        assert profile.experience_years == 5
        assert "ML Engineer" in profile.job_titles
        assert profile.raw_text == "Sample resume text"

    def test_retries_once_then_falls_back_llm_called_twice(self) -> None:
        """Test 2: on JSON parse failure, retries once, then falls back.
        Confirms LLM was called exactly twice."""
        bad_response = MagicMock()
        bad_response.content = "This is not valid JSON at all"

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = bad_response

        mock_template = MagicMock()
        mock_template.__or__ = MagicMock(return_value=mock_chain)

        with patch("agents.resume_analyzer_agent.get_primary_llm") as mock_get_llm, \
             patch("agents.resume_analyzer_agent.ChatPromptTemplate") as MockTemplate:
            mock_get_llm.return_value = MagicMock()
            MockTemplate.from_template.return_value = mock_template

            profile = extract_profile_from_text("Some resume text")

        # Should have tried exactly 2 times (MAX_RETRIES = 2)
        assert mock_chain.invoke.call_count == 2

        # Falls back to minimal profile with raw_text
        assert isinstance(profile, CandidateProfile)
        assert profile.raw_text == "Some resume text"


class TestResumeAnalyzerAgent:
    """Tests for the resume_analyzer_agent LangGraph node."""

    def test_returns_unchanged_when_resume_text_none_or_empty(
        self, minimal_state: AgentState
    ) -> None:
        """Test 3: state unchanged when resume_text is None or empty.
        Confirms no LLM call is made."""
        # Test with None
        state_none = {**minimal_state, "resume_text": None}

        with patch("agents.resume_analyzer_agent.get_primary_llm") as mock_llm:
            result_none = resume_analyzer_agent(state_none)
            mock_llm.assert_not_called()

        assert result_none["candidate_profile"] is None
        assert "resume_analyzer" in result_none["completed_agents"]

        # Test with empty string
        state_empty = {**minimal_state, "resume_text": ""}

        with patch("agents.resume_analyzer_agent.get_primary_llm") as mock_llm:
            result_empty = resume_analyzer_agent(state_empty)
            mock_llm.assert_not_called()

        assert result_empty["candidate_profile"] is None
        assert "resume_analyzer" in result_empty["completed_agents"]
