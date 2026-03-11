"""Job Scraper Agent — fetches and deduplicates listings from Adzuna, RemoteOK, and Jooble.

LangGraph node name: "job_scraper"
Reads from state: job_title, location, num_results
Writes to state: raw_jobs, jobs, scrape_summary, active_agent, completed_agents
Does NOT use LLM — pure API calls and data transformation.
"""

import os
import time
import uuid
from typing import Any, Dict, List, Tuple

import requests
from dotenv import load_dotenv

from utils.logger import log
from utils.state import AgentState, Job

load_dotenv()

# Config from environment
REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "15"))
MAX_JOBS_PER_SOURCE: int = int(os.getenv("MAX_JOBS_PER_SOURCE", "20"))
USER_AGENT: str = "JobAssistantBot/1.0 (portfolio project)"


def _sanitize_query(text: str) -> str:
    """Sanitize user input for use in API query parameters.

    Args:
        text: Raw user input string.

    Returns:
        Sanitized string with HTML stripped and length limited to 100 chars.
    """
    import re
    clean = re.sub(r"<[^>]+>", "", text)
    return clean.strip()[:100]


def fetch_adzuna(job_title: str, location: str, num_results: int) -> List[Dict[str, Any]]:
    """Fetch job listings from the Adzuna API.

    Args:
        job_title: Job title search query.
        location: Target location for job search.
        num_results: Maximum number of results to fetch.

    Returns:
        List of normalized job dictionaries. Empty list on failure.
    """
    app_id = os.getenv("ADZUNA_APP_ID", "")
    app_key = os.getenv("ADZUNA_APP_KEY", "")

    if not app_id or app_id == "your_adzuna_app_id" or not app_key or app_key == "your_adzuna_app_key":
        log.warning("adzuna_skipped", reason="ADZUNA_APP_ID or ADZUNA_APP_KEY not configured")
        return []

    country = os.getenv("ADZUNA_COUNTRY", "us")
    url = f"https://api.adzuna.com/v1/api/jobs/{country}/search/1"
    params = {
        "app_id": app_id,
        "app_key": app_key,
        "what": _sanitize_query(job_title),
        "where": _sanitize_query(location),
        "results_per_page": min(num_results, MAX_JOBS_PER_SOURCE),
    }

    try:
        session = requests.Session()
        session.headers.update({"User-Agent": USER_AGENT})
        resp = session.get(url, params=params, timeout=REQUEST_TIMEOUT)

        if resp.status_code >= 500:
            log.error("api_call_failed", source="adzuna", error="Server error", status_code=resp.status_code)
            return []

        if resp.status_code >= 400:
            log.error("api_call_failed", source="adzuna", error=resp.text[:200], status_code=resp.status_code)
            return []

        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])

        jobs: List[Dict[str, Any]] = []
        for item in results:
            salary = None
            sal_min = item.get("salary_min")
            sal_max = item.get("salary_max")
            if sal_min and sal_max:
                salary = f"${int(sal_min):,} – ${int(sal_max):,}"
            elif sal_min:
                salary = f"${int(sal_min):,}+"
            elif sal_max:
                salary = f"Up to ${int(sal_max):,}"

            jobs.append({
                "id": str(uuid.uuid4()),
                "title": item.get("title", "Unknown"),
                "company": item.get("company", {}).get("display_name", "Unknown"),
                "location": item.get("location", {}).get("display_name", "Unknown"),
                "description": (item.get("description", "")[:3000]),
                "url": item.get("redirect_url", ""),
                "source": "adzuna",
                "salary": salary,
                "tags": [],
            })

        log.info("api_call", source="adzuna", status="success", jobs_returned=len(jobs))
        return jobs

    except requests.exceptions.Timeout:
        log.warning("api_call_failed", source="adzuna", error="Request timed out")
        return []
    except requests.exceptions.RequestException as e:
        log.error("api_call_failed", source="adzuna", error=str(e))
        return []
    except Exception as e:
        log.error("api_call_failed", source="adzuna", error=str(e))
        return []


def fetch_remoteok(job_title: str) -> List[Dict[str, Any]]:
    """Fetch remote job listings from the RemoteOK API.

    Args:
        job_title: Job title search query for keyword filtering.

    Returns:
        List of normalized job dictionaries. Empty list on failure.
    """
    url = "https://remoteok.com/api"

    try:
        session = requests.Session()
        session.headers.update({"User-Agent": USER_AGENT})
        resp = session.get(url, timeout=REQUEST_TIMEOUT)

        if resp.status_code >= 500:
            log.error("api_call_failed", source="remoteok", error="Server error", status_code=resp.status_code)
            return []

        if resp.status_code >= 400:
            log.error("api_call_failed", source="remoteok", error=resp.text[:200], status_code=resp.status_code)
            return []

        resp.raise_for_status()
        data = resp.json()

        # Filter by keyword — check if any word from job_title appears in position or tags
        keywords = [w.lower() for w in _sanitize_query(job_title).split()]

        jobs: List[Dict[str, Any]] = []
        for item in data:
            # Skip legal notice (first element) — check for absence of "position" key
            if "position" not in item:
                continue

            position = item.get("position", "").lower()
            tags_list = [t.lower() for t in item.get("tags", [])]
            tag_str = " ".join(tags_list)

            # Check if any search keyword matches position or tags
            if not any(kw in position or kw in tag_str for kw in keywords):
                continue

            jobs.append({
                "id": str(uuid.uuid4()),
                "title": item.get("position", "Unknown"),
                "company": item.get("company", "Unknown"),
                "location": "Remote",
                "description": (item.get("description", "")[:3000]),
                "url": item.get("url", ""),
                "source": "remoteok",
                "salary": item.get("salary") if item.get("salary") else None,
                "tags": item.get("tags", []),
            })

            if len(jobs) >= MAX_JOBS_PER_SOURCE:
                break

        log.info("api_call", source="remoteok", status="success", jobs_returned=len(jobs))
        return jobs

    except requests.exceptions.Timeout:
        log.warning("api_call_failed", source="remoteok", error="Request timed out")
        return []
    except requests.exceptions.RequestException as e:
        log.error("api_call_failed", source="remoteok", error=str(e))
        return []
    except Exception as e:
        log.error("api_call_failed", source="remoteok", error=str(e))
        return []


def fetch_jooble(job_title: str, location: str, num_results: int) -> List[Dict[str, Any]]:
    """Fetch job listings from the Jooble API.

    Args:
        job_title: Job title search query.
        location: Target location for job search.
        num_results: Maximum number of results to fetch.

    Returns:
        List of normalized job dictionaries. Empty list on failure.
    """
    api_key = os.getenv("JOOBLE_API_KEY", "")

    if not api_key or api_key == "your_jooble_api_key":
        log.warning("jooble_skipped", reason="JOOBLE_API_KEY not configured")
        return []

    url = f"https://jooble.org/api/{api_key}"
    payload = {
        "keywords": _sanitize_query(job_title),
        "location": _sanitize_query(location),
        "resultonpage": min(num_results, MAX_JOBS_PER_SOURCE),
    }

    try:
        session = requests.Session()
        session.headers.update({"User-Agent": USER_AGENT})
        resp = session.post(url, json=payload, timeout=REQUEST_TIMEOUT)

        if resp.status_code >= 500:
            log.error("api_call_failed", source="jooble", error="Server error", status_code=resp.status_code)
            return []

        if resp.status_code >= 400:
            log.error("api_call_failed", source="jooble", error=resp.text[:200], status_code=resp.status_code)
            return []

        resp.raise_for_status()
        data = resp.json()
        results = data.get("jobs", [])

        jobs: List[Dict[str, Any]] = []
        for item in results:
            jobs.append({
                "id": str(uuid.uuid4()),
                "title": item.get("title", "Unknown"),
                "company": item.get("company", "Unknown"),
                "location": item.get("location", "Unknown"),
                "description": (item.get("snippet", "")[:3000]),
                "url": item.get("link", ""),
                "source": "jooble",
                "salary": item.get("salary") if item.get("salary") else None,
                "tags": [],
            })

        log.info("api_call", source="jooble", status="success", jobs_returned=len(jobs))
        return jobs

    except requests.exceptions.Timeout:
        log.warning("api_call_failed", source="jooble", error="Request timed out")
        return []
    except requests.exceptions.RequestException as e:
        log.error("api_call_failed", source="jooble", error=str(e))
        return []
    except Exception as e:
        log.error("api_call_failed", source="jooble", error=str(e))
        return []


def deduplicate_jobs(jobs: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    """Remove duplicate jobs based on (title, company) tuple.

    Args:
        jobs: List of job dictionaries to deduplicate.

    Returns:
        Tuple of (deduplicated jobs list, number of duplicates removed).
    """
    seen: set = set()
    unique_jobs: List[Dict[str, Any]] = []

    for job in jobs:
        key = (job["title"].lower().strip(), job["company"].lower().strip())
        if key not in seen:
            seen.add(key)
            unique_jobs.append(job)

    duplicates_removed = len(jobs) - len(unique_jobs)
    if duplicates_removed > 0:
        log.info("deduplication", duplicates_removed=duplicates_removed)

    return unique_jobs, duplicates_removed


def job_scraper_agent(state: AgentState) -> AgentState:
    """LangGraph node: fetches jobs from 3 APIs, deduplicates, and updates state.

    Reads job_title, location, and num_results from state. Queries Adzuna,
    RemoteOK, and Jooble in sequence. Each source is wrapped in its own
    error handler — a failure in one source does not stop the others.

    Args:
        state: Current pipeline state.

    Returns:
        Updated state with jobs, raw_jobs, and scrape_summary.
    """
    start_time = time.time()
    job_title: str = state["job_title"]
    location: str = state["location"]
    num_results: int = state.get("num_results", 10)

    log.info("agent_started", agent="job_scraper", job_title=job_title, location=location)

    # Fetch from all 3 sources independently
    adzuna_jobs = fetch_adzuna(job_title, location, num_results)
    remoteok_jobs = fetch_remoteok(job_title)
    jooble_jobs = fetch_jooble(job_title, location, num_results)

    # Combine all raw results
    all_raw_jobs = adzuna_jobs + remoteok_jobs + jooble_jobs

    # Deduplicate
    unique_job_dicts, duplicates_removed = deduplicate_jobs(all_raw_jobs)

    # Convert to Job Pydantic models
    jobs: List[Job] = []
    for job_dict in unique_job_dicts:
        try:
            jobs.append(Job(**job_dict))
        except Exception as e:
            log.warning("job_parse_error", error=str(e), job_title=job_dict.get("title", "unknown"))
            continue

    # Count sources that returned results
    sources_with_results: List[str] = []
    source_counts: Dict[str, int] = {
        "Adzuna": len(adzuna_jobs),
        "RemoteOK": len(remoteok_jobs),
        "Jooble": len(jooble_jobs),
    }
    for source_name, count in source_counts.items():
        if count > 0:
            sources_with_results.append(source_name)

    # Build summary
    source_detail = ", ".join(
        f"{name} ({count} jobs)" for name, count in source_counts.items()
    )
    scrape_summary = (
        f"Found {len(jobs)} unique jobs for '{job_title}' in '{location}' "
        f"across {len(sources_with_results)} source(s): {', '.join(sources_with_results) if sources_with_results else 'none'}.\n"
        f"Sources: {source_detail}.\n"
        f"{duplicates_removed} duplicates removed."
    )

    # Handle all-source failure
    error = state.get("error")
    if not jobs:
        error = (
            f"No jobs found for '{job_title}' in '{location}'. "
            "This could be due to missing API keys, network issues, or no matching results. "
            "Check your .env configuration and try different search terms."
        )

    elapsed_ms = int((time.time() - start_time) * 1000)
    log.info("agent_completed", agent="job_scraper", total_jobs=len(jobs), duration_ms=elapsed_ms)

    return {
        **state,
        "raw_jobs": all_raw_jobs,
        "jobs": jobs,
        "scrape_summary": scrape_summary,
        "error": error,
        "active_agent": None,
        "completed_agents": state.get("completed_agents", []) + ["job_scraper"],
    }
