"""Data models and shared state for the Multi-Agent Job Application Assistant.

This module defines all Pydantic models used for inter-agent data contracts
and the LangGraph AgentState TypedDict that serves as the shared state
across all pipeline nodes.
"""

from typing import List, Optional, TypedDict

from pydantic import BaseModel, Field


class Job(BaseModel):
    """A single job listing normalized from any API source.

    Attributes:
        id: UUID generated at scrape time.
        title: Job title as returned by the API.
        company: Company name.
        location: Location string (may be "Remote").
        description: Full job description text.
        url: Direct link to job posting.
        source: Origin API — "adzuna", "remoteok", or "jooble".
        salary: Formatted salary range string, e.g. "$80,000 – $120,000".
        tags: Tech stack tags, e.g. ["python", "aws", "docker"].
        fit_score: 0–100, filled by Fit Scorer Agent (v2).
        fit_reasoning: LLM explanation of score (v2).
    """

    id: str
    title: str
    company: str
    location: str
    description: str
    url: str
    source: str
    salary: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    fit_score: Optional[int] = Field(default=None, ge=0, le=100)
    fit_reasoning: Optional[str] = None


class CandidateProfile(BaseModel):
    """Structured profile extracted from an uploaded resume PDF.

    Attributes:
        name: Candidate's full name.
        email: Contact email address.
        skills: Technical skills list, e.g. ["Python", "LangChain", "AWS"].
        experience_years: Estimated total years of professional experience.
        job_titles: Past job titles held.
        education: Education entries, e.g. ["BSc Computer Science, University of X"].
        summary: LLM-generated 2-3 sentence professional summary.
        raw_text: Full extracted resume text for downstream LLM context.
    """

    name: Optional[str] = None
    email: Optional[str] = None
    skills: List[str] = Field(default_factory=list)
    experience_years: Optional[int] = None
    job_titles: List[str] = Field(default_factory=list)
    education: List[str] = Field(default_factory=list)
    summary: str = ""
    raw_text: str = ""


class AgentState(TypedDict, total=False):
    """LangGraph shared state — the data contract between all agents.

    All agents read from and write to this state dictionary.

    User inputs:
        job_title: Search query, e.g. "AI Engineer".
        location: Target location, e.g. "Remote" or "New York".
        num_results: Number of jobs to fetch per source.
        resume_text: Raw text extracted from uploaded PDF.

    Agent outputs:
        raw_jobs: Raw API responses before parsing.
        jobs: Parsed, deduplicated Job objects.
        scrape_summary: Human-readable summary of scrape results.
        candidate_profile: Structured profile from Resume Analyzer.
        ranked_jobs: Jobs sorted by fit_score descending (v2).
        tailored_bullets: job_id → rewritten bullets list (v2).
        cover_letters: job_id → cover letter string (v2).

    Pipeline metadata:
        error: Set by any agent on failure.
        active_agent: Name of currently running agent (for UI).
        completed_agents: Agents that have finished successfully.
    """

    # User inputs
    job_title: str
    location: str
    num_results: int
    resume_text: Optional[str]

    # Agent outputs
    raw_jobs: List[dict]
    jobs: List[Job]
    scrape_summary: str
    candidate_profile: Optional[CandidateProfile]
    ranked_jobs: List[Job]
    tailored_bullets: Optional[dict]
    cover_letters: Optional[dict]

    # Pipeline metadata
    error: Optional[str]
    active_agent: Optional[str]
    completed_agents: List[str]
