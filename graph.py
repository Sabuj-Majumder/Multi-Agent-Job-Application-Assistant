"""LangGraph pipeline definition — imports and wires all agents.

This module defines the directed graph that orchestrates all agents.
v1 scope: Job Scraper → (conditional) Resume Analyzer → END.
"""

from langgraph.graph import END, START, StateGraph

from agents.job_scraper_agent import job_scraper_agent
from agents.resume_analyzer_agent import resume_analyzer_agent
from utils.state import AgentState


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


def build_graph() -> StateGraph:
    """Build and compile the LangGraph agent pipeline.

    Creates the v1 graph with Job Scraper and Resume Analyzer nodes.
    Resume Analyzer only runs if resume_text is present in state
    (conditional edge).

    Returns:
        Compiled LangGraph StateGraph ready for invocation.
    """
    builder = StateGraph(AgentState)

    # Register agent nodes
    builder.add_node("job_scraper", job_scraper_agent)
    builder.add_node("resume_analyzer", resume_analyzer_agent)

    # Wire edges
    builder.add_edge(START, "job_scraper")
    builder.add_conditional_edges("job_scraper", should_run_resume_analyzer)
    builder.add_edge("resume_analyzer", END)

    return builder.compile()


# Module-level compiled pipeline — import and invoke this
pipeline = build_graph()
