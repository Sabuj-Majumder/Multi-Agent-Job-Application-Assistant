"""Resume Tailor Agent — rewrites resume bullets to match top jobs (v2).

LangGraph node name: "resume_tailor"
Reads from state: ranked_jobs (top 3), candidate_profile
Writes to state: tailored_bullets — dict of {job_id: [bullet_1, bullet_2, ...]}
Uses LLM: llama-3.3-70b-versatile

Status: v2 — not yet implemented.
"""

from utils.state import AgentState


def resume_tailor_agent(state: AgentState) -> AgentState:
    """Rewrite candidate resume bullets to better match top-ranked jobs.

    TODO: Implement v2 resume tailoring with the following logic:
        - Take top 3 jobs from state["ranked_jobs"]
        - For each job, call the primary LLM with RESUME_TAILOR_PROMPT
        - Extract existing bullet points from candidate_profile.raw_text
        - Rewrite 4-6 bullets per job using keywords from the JD
        - Never fabricate experience — only rephrase and emphasize
        - Store results in tailored_bullets dict

    Args:
        state: Current pipeline state with ranked_jobs and candidate_profile.

    Returns:
        Updated state with tailored_bullets populated.
    """
    pass  # TODO: Implement in v2
    return {
        **state,
        "tailored_bullets": None,
        "active_agent": None,
        "completed_agents": state.get("completed_agents", []) + ["resume_tailor"],
    }
