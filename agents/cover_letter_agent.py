"""Cover Letter Agent — generates personalized cover letters for top jobs (v2).

LangGraph node name: "cover_letter"
Reads from state: ranked_jobs (top 3), candidate_profile, tailored_bullets
Writes to state: cover_letters — dict of {job_id: cover_letter_string}
Uses LLM: llama-3.3-70b-versatile
"""

import time
from typing import Dict

import structlog
from langchain_core.prompts import ChatPromptTemplate

from utils.llm import get_primary_llm
from utils.prompts import COVER_LETTER_PROMPT
from utils.state import AgentState

log = structlog.get_logger()

def cover_letter_agent(state: AgentState) -> AgentState:
    """Write a personalized cover letter for each of the top ranked jobs.
    
    Args:
        state: Current pipeline state with ranked_jobs, candidate_profile, and tailored_bullets.
        
    Returns:
        Updated state with cover_letters populated.
    """
    log.info("agent_started", agent="cover_letter")
    start_time = time.time()

    ranked_jobs = state.get("ranked_jobs", [])
    candidate_profile = state.get("candidate_profile")
    tailored_bullets = state.get("tailored_bullets") or {}

    if not ranked_jobs or candidate_profile is None:
        log.warning(
            "cover_letter_skipped",
            reason="Missing ranked_jobs or candidate_profile",
        )
        return {
            **state,
            "active_agent": None,
            "completed_agents": state.get("completed_agents", []) + ["cover_letter"],
        }

    top_jobs = ranked_jobs[:3]

    if hasattr(candidate_profile, "name"):
        c_name = getattr(candidate_profile, "name", "")
        c_skills = getattr(candidate_profile, "skills", [])
        c_exp = getattr(candidate_profile, "experience_years", "")
        c_titles = getattr(candidate_profile, "job_titles", [])
    else:
        c_name = candidate_profile.get("name", "")
        c_skills = candidate_profile.get("skills", [])
        c_exp = candidate_profile.get("experience_years", "")
        c_titles = candidate_profile.get("job_titles", [])

    llm = get_primary_llm()
    prompt = ChatPromptTemplate.from_template(COVER_LETTER_PROMPT)
    chain = prompt | llm

    cover_letters: Dict[str, str] = {}
    success_count = 0

    for job in top_jobs:
        if hasattr(job, "id"):
            job_id = job.id
            j_title = job.title
            j_company = job.company
            j_desc = job.description
        else:
            job_id = job.get("id", "")
            j_title = job.get("title", "Unknown")
            j_company = job.get("company", "Unknown")
            j_desc = job.get("description", "")
        
        job_bullets = tailored_bullets.get(job_id, [])
        bullets_text = "\n".join([f"- {b}" for b in job_bullets]) if job_bullets else "None"

        try:
            response = chain.invoke(
                {
                    "job_title": j_title,
                    "company": j_company,
                    "job_description": j_desc[:2000],
                    "candidate_name": c_name or "Candidate",
                    "skills": ", ".join(c_skills) if c_skills else "Various technical skills",
                    "experience_years": c_exp if c_exp is not None else "several",
                    "job_titles": ", ".join(c_titles) if c_titles else "software engineering roles",
                    "tailored_bullets": bullets_text,
                }
            )
            
            raw_content = response.content
            if not isinstance(raw_content, str):
                raw_content = str(raw_content)

            cover_letters[job_id] = raw_content.strip()
            success_count += 1
            log.info("cover_letter_generated", job_id=job_id, status="success")
        except Exception as e:
            log.error(
                "cover_letter_failed",
                job_id=job_id,
                error=str(e),
            )
            cover_letters[job_id] = ""

    elapsed_ms = int((time.time() - start_time) * 1000)
    log.info(
        "agent_completed",
        agent="cover_letter",
        total_generated=success_count,
        duration_ms=elapsed_ms,
    )

    return {
        **state,
        "cover_letters": cover_letters,
        "active_agent": None,
        "completed_agents": state.get("completed_agents", []) + ["cover_letter"],
    }
