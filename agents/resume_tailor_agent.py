"""Resume Tailor Agent — rewrites resume bullets to match top jobs.

LangGraph node name: "resume_tailor"
Reads from state: ranked_jobs (top 3), candidate_profile
Writes to state: tailored_bullets — dict of {job_id: [bullet_1, bullet_2, ...]}
Uses LLM: llama-3.3-70b-versatile
"""

import json
import time
from typing import Dict, List, cast

import structlog
from langchain_core.prompts import ChatPromptTemplate

from utils.llm import get_primary_llm
from utils.prompts import RESUME_TAILOR_PROMPT
from utils.state import AgentState, Job, CandidateProfile

log = structlog.get_logger()


def clean_json(content: str) -> str:
    """Strip markdown fences from LLM output.
    
    Args:
        content: Raw string response from LLM.
        
    Returns:
        Clean JSON string.
    """
    cleaned = content.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()


def resume_tailor_agent(state: AgentState) -> AgentState:
    """Rewrite candidate resume bullets to better match top-ranked jobs.
    
    Args:
        state: Current pipeline state with ranked_jobs and candidate_profile.
        
    Returns:
        Updated state with tailored_bullets populated.
    """
    job_title_query = state.get("job_title", "unknown")
    log.info("agent_started", agent="resume_tailor", query=job_title_query)
    start_time = time.time()

    ranked_jobs = state.get("ranked_jobs", [])
    candidate_profile = state.get("candidate_profile")

    if not ranked_jobs or not candidate_profile:
        log.warning(
            "resume_tailor_skipped",
            reason="Missing ranked_jobs or candidate_profile",
        )
        return {
            **state,
            "active_agent": None,
            "completed_agents": state.get("completed_agents", []) + ["resume_tailor"],
        }

    # Take top 3
    top_jobs = ranked_jobs[:3]
    
    # We must treat the profile appropriately based on whether it is a dict or a Pydantic model
    if hasattr(candidate_profile, "raw_text"):
        resume_raw_text = candidate_profile.raw_text[:3000]
    else:
        resume_raw_text = candidate_profile.get("raw_text", "")[:3000]

    llm = get_primary_llm()
    prompt = ChatPromptTemplate.from_template(RESUME_TAILOR_PROMPT)
    chain = prompt | llm

    tailored_bullets: Dict[str, List[str]] = {}

    for job in top_jobs:
        if isinstance(job, Job):
            job_id = job.id
            j_title = job.title
            j_company = job.company
            j_desc = job.description
        else:
            job_id = job.get("id", "unknown")
            j_title = job.get("title", "")
            j_company = job.get("company", "")
            j_desc = job.get("description", "")

        try:
            response = chain.invoke(
                {
                    "job_title": j_title,
                    "company": j_company,
                    "job_description": j_desc[:2000],
                    "resume_text": resume_raw_text,
                }
            )
            
            raw_content = response.content
            if not isinstance(raw_content, str):
                raw_content = str(raw_content)

            parsed_json = json.loads(clean_json(raw_content))
            
            if isinstance(parsed_json, list) and all(isinstance(i, str) for i in parsed_json):
                 tailored_bullets[job_id] = parsed_json
            elif isinstance(parsed_json, dict) and "bullets" in parsed_json and isinstance(parsed_json["bullets"], list):
                 tailored_bullets[job_id] = parsed_json["bullets"]
            else:
                 raise ValueError("Output is not a valid list of strings.")
            log.info("job_tailored", job_id=job_id, status="success")
        except Exception as e:
            log.error(
                "job_tailor_failed",
                job_id=job_id,
                error=str(e),
                status="parse_error",
            )
            tailored_bullets[job_id] = []

    elapsed_ms = int((time.time() - start_time) * 1000)
    log.info(
        "agent_completed",
        agent="resume_tailor",
        total_jobs_tailored=len(top_jobs),
        duration_ms=elapsed_ms,
    )

    return {
        **state,
        "tailored_bullets": tailored_bullets,
        "active_agent": None,
        "completed_agents": state.get("completed_agents", []) + ["resume_tailor"],
    }
