"""Tests for the Job Scraper Agent with mocked HTTP.

Test cases:
1. fetch_remoteok returns correct structure with mocked HTTP
2. fetch_adzuna gracefully skips when env key is missing
3. deduplicate correctly removes jobs with same title+company
4. job_scraper_agent node writes correct keys to state
5. job_scraper_agent handles all-source failure gracefully
"""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from agents.job_scraper_agent import (
    deduplicate_jobs,
    fetch_adzuna,
    fetch_jooble,
    fetch_remoteok,
    job_scraper_agent,
)
from utils.state import AgentState


class TestFetchRemoteOK:
    """Tests for the RemoteOK API fetcher."""

    def test_fetch_remoteok_success(self) -> None:
        """fetch_remoteok should return correct structure with mocked HTTP."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"legal": "notice"},  # First element is the legal notice — skipped
            {
                "id": "123",
                "position": "AI Engineer",
                "company": "Acme Corp",
                "description": "Build AI systems using Python and TensorFlow",
                "url": "https://remoteok.com/jobs/123",
                "tags": ["python", "ai", "tensorflow"],
                "salary": "$100,000 - $150,000",
            },
            {
                "id": "456",
                "position": "Data Analyst",
                "company": "DataCo",
                "description": "Analyze business data",
                "url": "https://remoteok.com/jobs/456",
                "tags": ["sql", "excel"],
                "salary": None,
            },
        ]
        mock_response.raise_for_status.return_value = None

        with patch("agents.job_scraper_agent.requests.Session") as MockSession:
            mock_session = MagicMock()
            mock_session.get.return_value = mock_response
            MockSession.return_value = mock_session

            result = fetch_remoteok("AI Engineer")

            assert len(result) == 1  # Only AI Engineer matches keyword
            assert result[0]["title"] == "AI Engineer"
            assert result[0]["company"] == "Acme Corp"
            assert result[0]["source"] == "remoteok"
            assert result[0]["location"] == "Remote"
            assert "python" in result[0]["tags"]

    def test_fetch_remoteok_no_matches(self) -> None:
        """fetch_remoteok should return empty list when no jobs match keyword."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"legal": "notice"},
            {
                "id": "789",
                "position": "Plumber",
                "company": "FixIt Inc",
                "description": "Fix pipes",
                "url": "https://remoteok.com/jobs/789",
                "tags": ["plumbing"],
            },
        ]
        mock_response.raise_for_status.return_value = None

        with patch("agents.job_scraper_agent.requests.Session") as MockSession:
            mock_session = MagicMock()
            mock_session.get.return_value = mock_response
            MockSession.return_value = mock_session

            result = fetch_remoteok("AI Engineer")
            assert len(result) == 0

    def test_fetch_remoteok_timeout(self) -> None:
        """fetch_remoteok should return empty list on timeout."""
        import requests as req_lib

        with patch("agents.job_scraper_agent.requests.Session") as MockSession:
            mock_session = MagicMock()
            mock_session.get.side_effect = req_lib.exceptions.Timeout("Connection timed out")
            MockSession.return_value = mock_session

            result = fetch_remoteok("AI Engineer")
            assert result == []


class TestFetchAdzuna:
    """Tests for the Adzuna API fetcher."""

    def test_fetch_adzuna_skips_when_key_missing(self) -> None:
        """fetch_adzuna should gracefully skip when env key is missing."""
        with patch.dict(os.environ, {"ADZUNA_APP_ID": "", "ADZUNA_APP_KEY": ""}, clear=False):
            result = fetch_adzuna("AI Engineer", "Remote", 10)
            assert result == []

    def test_fetch_adzuna_skips_placeholder_key(self) -> None:
        """fetch_adzuna should skip when env key is still the placeholder."""
        with patch.dict(
            os.environ,
            {"ADZUNA_APP_ID": "your_adzuna_app_id", "ADZUNA_APP_KEY": "your_adzuna_app_key"},
            clear=False,
        ):
            result = fetch_adzuna("AI Engineer", "Remote", 10)
            assert result == []

    def test_fetch_adzuna_success(self) -> None:
        """fetch_adzuna should return correctly parsed jobs on success."""
        # Load mock response
        fixtures_path = os.path.join(os.path.dirname(__file__), "fixtures", "mock_adzuna_response.json")
        with open(fixtures_path) as f:
            mock_data = json.load(f)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_data
        mock_response.raise_for_status.return_value = None

        with patch.dict(
            os.environ,
            {"ADZUNA_APP_ID": "test_id", "ADZUNA_APP_KEY": "test_key"},
            clear=False,
        ):
            with patch("agents.job_scraper_agent.requests.Session") as MockSession:
                mock_session = MagicMock()
                mock_session.get.return_value = mock_response
                MockSession.return_value = mock_session

                result = fetch_adzuna("AI Engineer", "Remote", 10)

                assert len(result) == 2
                assert result[0]["title"] == "AI Engineer"
                assert result[0]["company"] == "TechCorp"
                assert result[0]["source"] == "adzuna"
                assert result[0]["salary"] == "$80,000 – $120,000"


class TestDeduplication:
    """Tests for job deduplication logic."""

    def test_deduplicate_removes_duplicates(self) -> None:
        """deduplicate should remove jobs with same title+company (case-insensitive)."""
        jobs = [
            {"title": "AI Engineer", "company": "TechCorp", "source": "adzuna"},
            {"title": "ai engineer", "company": "techcorp", "source": "remoteok"},
            {"title": "Data Scientist", "company": "DataCo", "source": "jooble"},
            {"title": "AI Engineer", "company": "OtherCo", "source": "adzuna"},
        ]

        unique_jobs, duplicates_removed = deduplicate_jobs(jobs)

        assert len(unique_jobs) == 3
        assert duplicates_removed == 1

    def test_deduplicate_no_duplicates(self) -> None:
        """deduplicate should return all jobs when no duplicates exist."""
        jobs = [
            {"title": "AI Engineer", "company": "TechCorp", "source": "adzuna"},
            {"title": "Data Scientist", "company": "DataCo", "source": "jooble"},
        ]

        unique_jobs, duplicates_removed = deduplicate_jobs(jobs)

        assert len(unique_jobs) == 2
        assert duplicates_removed == 0

    def test_deduplicate_empty_list(self) -> None:
        """deduplicate should handle empty list."""
        unique_jobs, duplicates_removed = deduplicate_jobs([])
        assert len(unique_jobs) == 0
        assert duplicates_removed == 0


class TestJobScraperAgent:
    """Tests for the job_scraper_agent LangGraph node function."""

    def test_agent_writes_correct_keys_to_state(self) -> None:
        """job_scraper_agent should write raw_jobs, jobs, scrape_summary to state."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"legal": "notice"},
            {
                "id": "1",
                "position": "AI Engineer",
                "company": "TestCo",
                "description": "AI work",
                "url": "https://example.com/1",
                "tags": ["python"],
            },
        ]
        mock_response.raise_for_status.return_value = None

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
            "active_agent": "job_scraper",
            "completed_agents": [],
        }

        with patch.dict(os.environ, {"ADZUNA_APP_ID": "", "JOOBLE_API_KEY": ""}, clear=False):
            with patch("agents.job_scraper_agent.requests.Session") as MockSession:
                mock_session = MagicMock()
                mock_session.get.return_value = mock_response
                mock_session.post.return_value = mock_response
                MockSession.return_value = mock_session

                result = job_scraper_agent(state)

                assert "jobs" in result
                assert "raw_jobs" in result
                assert "scrape_summary" in result
                assert "job_scraper" in result["completed_agents"]
                assert result["active_agent"] is None

    def test_agent_handles_all_source_failure(self) -> None:
        """job_scraper_agent should set error when all sources fail."""
        import requests as req_lib

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
            "active_agent": "job_scraper",
            "completed_agents": [],
        }

        with patch.dict(os.environ, {"ADZUNA_APP_ID": "", "JOOBLE_API_KEY": ""}, clear=False):
            with patch("agents.job_scraper_agent.requests.Session") as MockSession:
                mock_session = MagicMock()
                mock_session.get.side_effect = req_lib.exceptions.Timeout("timeout")
                mock_session.post.side_effect = req_lib.exceptions.Timeout("timeout")
                MockSession.return_value = mock_session

                result = job_scraper_agent(state)

                assert result["error"] is not None
                assert len(result["jobs"]) == 0
                assert "job_scraper" in result["completed_agents"]
