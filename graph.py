"""LangGraph pipeline definition — imports and wires all agents.

This module defines the directed graph that orchestrates all agents.
Graph: Job Scraper → (conditional) Resume Analyzer → (conditional) Fit Scorer → END.
"""

from langgraph.graph import END, START, StateGraph

from agents.fit_scorer_agent import fit_scorer_agent
from agents.job_scraper_agent import job_scraper_agent
from agents.resume_analyzer_agent import resume_analyzer_agent
from agents.resume_tailor_agent import resume_tailor_agent
from agents.cover_letter_agent import cover_letter_agent
from utils.state import AgentState
from utils.llm import setup_tracing
from langsmith import traceable


def should_run_resume_analyzer(state: AgentState) -> str:
    """Conditional edge: only analyze resume if text was provided.

    Args:
        state: Current pipeline state.

    Returns:
        "resume_analyzer" if resume_text is present, END otherwise.
    """
    if state.get("resume_text"):
        return "resume_analyzer"
    return END


def should_run_fit_scorer(state: AgentState) -> str:
    """Conditional edge: only score fit if candidate_profile exists.

    Args:
        state: Current pipeline state.

    Returns:
        "fit_scorer" if candidate_profile exists, END otherwise.
    """
    if state.get("candidate_profile"):
        return "fit_scorer"
    return END


def should_run_resume_tailor(state: AgentState) -> str:
    """Conditional edge: only configure resume tailor if ranked_jobs exist.
    
    Args:
        state: Current pipeline state.
        
    Returns:
        "resume_tailor" if ranked_jobs exists, END otherwise.
    """
    if state.get("ranked_jobs"):
        return "resume_tailor"
    return END


def should_run_cover_letter(state: AgentState) -> str:
    """Conditional edge: only write cover letter if tailored_bullets exist.
    
    Args:
        state: Current pipeline state.
        
    Returns:
        "cover_letter" if tailored_bullets exists, END otherwise.
    """
    if state.get("tailored_bullets"):
        return "cover_letter"
    return END


def build_graph() -> StateGraph:
    """Build and compile the LangGraph agent pipeline.

    Creates the graph with Job Scraper, Resume Analyzer, and Fit Scorer nodes.
    Resume Analyzer only runs if resume_text is present in state.
    Fit Scorer only runs if candidate_profile exists after resume analysis.

    Returns:
        Compiled LangGraph StateGraph ready for invocation.
    """
    builder = StateGraph(AgentState)

    # Register agent nodes
    builder.add_node("job_scraper", job_scraper_agent)
    builder.add_node("resume_analyzer", resume_analyzer_agent)
    builder.add_node("fit_scorer", fit_scorer_agent)
    builder.add_node("resume_tailor", resume_tailor_agent)
    builder.add_node("cover_letter", cover_letter_agent)

    # Wire edges
    builder.add_edge(START, "job_scraper")
    builder.add_conditional_edges("job_scraper", should_run_resume_analyzer)
    builder.add_conditional_edges("resume_analyzer", should_run_fit_scorer)
    builder.add_conditional_edges("fit_scorer", should_run_resume_tailor)
    builder.add_conditional_edges("resume_tailor", should_run_cover_letter)
    builder.add_edge("cover_letter", END)

    return builder.compile()


# Module-level compiled pipeline — import and invoke this
pipeline = build_graph()

@traceable(name="job-assistant-pipeline")
def run_pipeline(state):
    """Traced wrapper around the full LangGraph pipeline."""
    return pipeline.invoke(state)
