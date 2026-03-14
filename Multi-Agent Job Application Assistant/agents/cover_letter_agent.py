"""Cover Letter Agent — generates personalized cover letters per job (v2).

LangGraph node name: "cover_letter"
Reads from state: ranked_jobs (top 3), candidate_profile, tailored_bullets
Writes to state: cover_letters — dict of {job_id: cover_letter_string}
Uses LLM: llama-3.3-70b-versatile

Status: v2 — not yet implemented.
"""

from utils.state import AgentState


def cover_letter_agent(state: AgentState) -> AgentState:
    """Write personalized cover letters for top-ranked job listings.

    TODO: Implement v2 cover letter generation with the following logic:
        - Take top 3 jobs from state["ranked_jobs"]
        - For each job, call the primary LLM with COVER_LETTER_PROMPT
        - Include candidate profile, job details, and tailored bullets as context
        - Generate a 3-paragraph, ~250-word cover letter per job
        - Format: Opening (why this role), Body (relevant experience), Closing (call to action)
        - Store results in cover_letters dict

    Args:
        state: Current pipeline state with ranked_jobs, candidate_profile, and tailored_bullets.

    Returns:
        Updated state with cover_letters populated.
    """
    pass  # TODO: Implement in v2
    return {
        **state,
        "cover_letters": None,
        "active_agent": None,
        "completed_agents": state.get("completed_agents", []) + ["cover_letter"],
    }
