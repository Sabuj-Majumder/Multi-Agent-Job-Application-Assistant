"""Fit Scorer Agent — scores job-candidate fit 0–100 for each listing.

LangGraph node name: "fit_scorer"
Reads from state: jobs, candidate_profile
Writes to state: ranked_jobs (sorted by fit_score descending), updates fit_score/fit_reasoning on each Job
Uses LLM: llama-3.1-8b-instant (fast model, called once per job)
"""

import json
import re
import time
from typing import Any, Dict, List, Optional

from langchain_core.prompts import ChatPromptTemplate

from utils.llm import get_fast_llm
from utils.logger import log
from utils.prompts import FIT_SCORER_PROMPT
from utils.state import AgentState, CandidateProfile, Job


def clean_json_response(text: str) -> str:
    """Strip markdown fences and extra whitespace from LLM JSON output.

    Args:
        text: Raw LLM response string that may contain markdown code fences.

    Returns:
        Cleaned string ready for JSON parsing.
    """
    cleaned = re.sub(r"```(?:json)?\s*", "", text)
    cleaned = re.sub(r"```", "", cleaned)
    return cleaned.strip()


def parse_score_response(response_text: str) -> Dict[str, Any]:
    """Parse LLM scoring response into a dictionary with score and reasoning.

    Args:
        response_text: Raw text from LLM response.

    Returns:
        Dictionary with 'score' (int) and 'reasoning' (str) keys.

    Raises:
        json.JSONDecodeError: If the response is not valid JSON after cleaning.
        KeyError: If required keys are missing.
        ValueError: If score is not a valid integer.
    """
    cleaned = clean_json_response(response_text)
    parsed = json.loads(cleaned)
    score = int(parsed["score"])
    reasoning = str(parsed["reasoning"])
    # Clamp score to 0–100 range
    score = max(0, min(100, score))
    return {"score": score, "reasoning": reasoning}


def score_single_job(
    job: Job,
    candidate_profile: CandidateProfile,
    llm: Any,
    prompt_template: ChatPromptTemplate,
) -> Dict[str, Any]:
    """Score a single job against the candidate profile using the LLM.

    Args:
        job: The Job object to score.
        candidate_profile: The candidate's structured profile.
        llm: The LLM instance to call.
        prompt_template: The prompt template for fit scoring.

    Returns:
        Dictionary with 'score' (int) and 'reasoning' (str).
    """
    skills_str = ", ".join(candidate_profile.skills) if candidate_profile.skills else "Not specified"
    experience_years = candidate_profile.experience_years if candidate_profile.experience_years is not None else "Not specified"
    past_titles = ", ".join(candidate_profile.job_titles) if candidate_profile.job_titles else "Not specified"
    description_truncated = job.description[:1500] if job.description else ""

    chain = prompt_template | llm
    response = chain.invoke({
        "job_title": job.title,
        "company": job.company,
        "description": description_truncated,
        "skills": skills_str,
        "experience_years": experience_years,
        "job_titles": past_titles,
    })

    return parse_score_response(response.content)


def fit_scorer_agent(state: AgentState) -> AgentState:
    """Score each job listing 0–100 based on candidate profile fit.

    Reads jobs and candidate_profile from state. If candidate_profile
    is None or empty, skips scoring and returns state unchanged with
    a log warning.

    For each job, calls the fast Groq LLM (llama-3.1-8b-instant) with
    the fit scoring prompt. On JSON parse failure, assigns score=0 and
    reasoning="Could not score" — never crashes.

    After scoring, sorts jobs by fit_score descending and writes to
    state["ranked_jobs"]. Also updates fit_score and fit_reasoning on
    each Job object in state["jobs"].

    Args:
        state: Current pipeline state with jobs and candidate_profile.

    Returns:
        Updated state with ranked_jobs populated and fit scores set.
    """
    start_time = time.time()
    log.info("agent_started", agent="fit_scorer")

    jobs: List[Job] = state.get("jobs", [])
    candidate_profile: Optional[CandidateProfile] = state.get("candidate_profile")

    # Skip scoring if no candidate profile is available
    if candidate_profile is None:
        log.warning("fit_scorer_skipped", reason="No candidate profile available")
        return {
            **state,
            "ranked_jobs": list(jobs),
            "active_agent": None,
            "completed_agents": state.get("completed_agents", []) + ["fit_scorer"],
        }

    # Also check for empty profile (no useful data)
    if not candidate_profile.skills and not candidate_profile.job_titles:
        log.warning(
            "fit_scorer_skipped",
            reason="Candidate profile is empty — no skills or job titles to score against",
        )
        return {
            **state,
            "ranked_jobs": list(jobs),
            "active_agent": None,
            "completed_agents": state.get("completed_agents", []) + ["fit_scorer"],
        }

    if not jobs:
        log.warning("fit_scorer_skipped", reason="No jobs to score")
        return {
            **state,
            "ranked_jobs": [],
            "active_agent": None,
            "completed_agents": state.get("completed_agents", []) + ["fit_scorer"],
        }

    # Initialize LLM and prompt template
    try:
        llm = get_fast_llm(temperature=0.1)
    except ValueError as e:
        log.error("fit_scorer_llm_init_failed", error=str(e))
        return {
            **state,
            "ranked_jobs": list(jobs),
            "error": f"Fit scoring failed: {str(e)}",
            "active_agent": None,
            "completed_agents": state.get("completed_agents", []) + ["fit_scorer"],
        }

    prompt_template = ChatPromptTemplate.from_template(FIT_SCORER_PROMPT)

    # Score each job sequentially
    scored_jobs: List[Job] = []
    for i, job in enumerate(jobs):
        try:
            result = score_single_job(job, candidate_profile, llm, prompt_template)
            # Update the Job object with score and reasoning
            job_dict = job.model_dump()
            job_dict["fit_score"] = result["score"]
            job_dict["fit_reasoning"] = result["reasoning"]
            scored_job = Job(**job_dict)
            scored_jobs.append(scored_job)

            log.info(
                "job_scored",
                agent="fit_scorer",
                job_title=job.title,
                company=job.company,
                fit_score=result["score"],
                job_index=i + 1,
                total_jobs=len(jobs),
            )

        except Exception as e:
            log.error(
                "job_scoring_failed",
                agent="fit_scorer",
                job_title=job.title,
                company=job.company,
                error=str(e),
                job_index=i + 1,
            )
            # On failure: assign score=0, reasoning="Could not score", continue
            job_dict = job.model_dump()
            job_dict["fit_score"] = 0
            job_dict["fit_reasoning"] = "Could not score"
            scored_job = Job(**job_dict)
            scored_jobs.append(scored_job)

    # Sort by fit_score descending
    ranked_jobs = sorted(scored_jobs, key=lambda j: j.fit_score or 0, reverse=True)

    elapsed_ms = int((time.time() - start_time) * 1000)
    log.info(
        "agent_completed",
        agent="fit_scorer",
        total_jobs_scored=len(scored_jobs),
        duration_ms=elapsed_ms,
    )

    return {
        **state,
        "jobs": scored_jobs,
        "ranked_jobs": ranked_jobs,
        "active_agent": None,
        "completed_agents": state.get("completed_agents", []) + ["fit_scorer"],
    }
