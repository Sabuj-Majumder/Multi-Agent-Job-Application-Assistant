"""Fit Scorer Agent — scores job-candidate fit 0–100 for each listing (v2).

LangGraph node name: "fit_scorer"
Reads from state: jobs, candidate_profile
Writes to state: ranked_jobs
Uses LLM: llama-3.1-8b-instant (fast model, called once per job)

Status: v2 — not yet implemented.
"""

from utils.state import AgentState


def fit_scorer_agent(state: AgentState) -> AgentState:
    """Score each job listing 0–100 based on candidate profile fit.

    TODO: Implement v2 fit scoring with the following logic:
        - Loop over all jobs in state["jobs"]
        - For each job, call the fast LLM with FIT_SCORER_PROMPT
        - Parse the JSON response for score (0-100) and reasoning
        - Add fit_score and fit_reasoning to each Job object
        - Sort jobs by fit_score descending → write to ranked_jobs

    Args:
        state: Current pipeline state with jobs and candidate_profile.

    Returns:
        Updated state with ranked_jobs populated.
    """
    pass  # TODO: Implement in v2
    return {
        **state,
        "ranked_jobs": state.get("jobs", []),
        "active_agent": None,
        "completed_agents": state.get("completed_agents", []) + ["fit_scorer"],
    }
