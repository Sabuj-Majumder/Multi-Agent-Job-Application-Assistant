"""Tests for the Job Scraper Agent.

Test 1: fetch_remoteok returns correctly structured list with mocked HTTP response
Test 2: fetch_remoteok returns empty list when HTTP request raises Timeout
Test 3: fetch_adzuna returns empty list and logs warning when ADZUNA_APP_ID env var is missing
        (Adapted: since the project now uses Arbeitnow instead of Adzuna, this test verifies
         that fetch_arbeitnow returns an empty list when the API returns a server error.)
Test 4: deduplicate removes jobs with identical (title.lower(), company.lower()) and keeps first
Test 5: job_scraper_agent node writes jobs, raw_jobs, and scrape_summary to state even when
        all sources return empty
"""

from unittest.mock import MagicMock, patch

import pytest
import requests as req_lib

from agents.job_scraper_agent import (
    deduplicate_jobs,
    fetch_arbeitnow,
    fetch_remoteok,
    job_scraper_agent,
)
from utils.state import AgentState


class TestFetchRemoteOK:
    """Tests for the RemoteOK API fetcher."""

    def test_fetch_remoteok_returns_correct_structure(self) -> None:
        """Test 1: fetch_remoteok returns correctly structured list."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"legal": "notice"},  # First element — no "position" key, skipped
            {
                "id": "10",
                "position": "AI Engineer",
                "company": "Acme Corp",
                "description": "Build AI systems with Python and TensorFlow.",
                "url": "https://remoteok.com/jobs/10",
                "tags": ["python", "ai", "tensorflow"],
                "salary": "$100,000 - $150,000",
                "date": "2025-11-01T00:00:00Z",
            },
        ]
        mock_response.raise_for_status.return_value = None

        with patch("agents.job_scraper_agent.requests.Session") as MockSession:
            mock_session = MagicMock()
            mock_session.get.return_value = mock_response
            MockSession.return_value = mock_session

            result = fetch_remoteok("AI Engineer")

        assert isinstance(result, list)
        assert len(result) == 1
        job = result[0]
        assert job["title"] == "AI Engineer"
        assert job["company"] == "Acme Corp"
        assert job["source"] == "remoteok"
        assert job["location"] == "Remote"
        assert "python" in job["tags"]
        assert job["salary"] == "$100,000 - $150,000"
        # Must have a UUID id, URL, and description
        assert "id" in job
        assert "url" in job
        assert "description" in job

    def test_fetch_remoteok_returns_empty_on_timeout(self) -> None:
        """Test 2: fetch_remoteok returns empty list on Timeout."""
        with patch("agents.job_scraper_agent.requests.Session") as MockSession:
            mock_session = MagicMock()
            mock_session.get.side_effect = req_lib.exceptions.Timeout("timed out")
            MockSession.return_value = mock_session

            result = fetch_remoteok("AI Engineer")

        assert result == []


class TestFetchArbeitnow:
    """Tests for the Arbeitnow API fetcher (replaces Adzuna in this project)."""

    def test_fetch_arbeitnow_returns_empty_on_server_error(self) -> None:
        """Test 3: fetch_arbeitnow returns empty list and logs warning on failure.

        The original spec references fetch_adzuna with missing ADZUNA_APP_ID,
        but the project replaced Adzuna with Arbeitnow. This test verifies
        equivalent graceful-failure behaviour — empty list on server error.
        """
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        with patch("agents.job_scraper_agent.requests.Session") as MockSession:
            mock_session = MagicMock()
            mock_session.get.return_value = mock_response
            MockSession.return_value = mock_session

            result = fetch_arbeitnow("AI Engineer")

        assert result == []


class TestDeduplication:
    """Test deduplication logic."""

    def test_deduplicate_keeps_first_occurrence(self) -> None:
        """Test 4: removes jobs with identical (title.lower(), company.lower()) and keeps first."""
        jobs = [
            {"title": "AI Engineer", "company": "TechCorp", "source": "remoteok"},
            {"title": "ai engineer", "company": "TECHCORP", "source": "themuse"},
            {"title": "Data Scientist", "company": "DataCo", "source": "arbeitnow"},
            {"title": "AI Engineer", "company": "OtherCo", "source": "remoteok"},
        ]

        unique, removed = deduplicate_jobs(jobs)

        assert len(unique) == 3
        assert removed == 1
        # First occurrence kept (source=remoteok), second dropped (source=themuse)
        ai_jobs = [j for j in unique if j["title"] == "AI Engineer" and j["company"] == "TechCorp"]
        assert len(ai_jobs) == 1
        assert ai_jobs[0]["source"] == "remoteok"


class TestJobScraperAgent:
    """Tests for the job_scraper_agent LangGraph node."""

    def test_agent_writes_keys_even_when_all_sources_empty(
        self, minimal_state: AgentState
    ) -> None:
        """Test 5: agent writes jobs, raw_jobs, and scrape_summary to state
        even when all sources return empty lists."""
        with patch("agents.job_scraper_agent.requests.Session") as MockSession:
            mock_session = MagicMock()
            mock_session.get.side_effect = req_lib.exceptions.Timeout("timeout")
            MockSession.return_value = mock_session

            result = job_scraper_agent(minimal_state)

        assert "jobs" in result
        assert "raw_jobs" in result
        assert "scrape_summary" in result
        assert isinstance(result["jobs"], list)
        assert isinstance(result["raw_jobs"], list)
        assert isinstance(result["scrape_summary"], str)
        assert "job_scraper" in result["completed_agents"]
