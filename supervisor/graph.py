"""LangGraph workflow construction for the HR supervisor system."""

import logging
from typing import Literal

from langgraph.graph import StateGraph, START, END

from .state import CandidateState
from .supervisor import supervisor_agent, members
from .agents import (
    analyze_resume,
    analyze_linkedin,
    analyze_cover_letter,
    match_candidates,
    rank_candidates,
)

logger = logging.getLogger(__name__)


def create_graph():
    """Create and compile the LangGraph workflow.
    
    Returns:
        Compiled graph ready for execution
    """
    # Create the graph
    workflow = StateGraph(CandidateState)
    
    # Add agent nodes
    workflow.add_node("ResumeAnalyzer", analyze_resume)
    workflow.add_node("LinkedInAnalyzer", analyze_linkedin)
    workflow.add_node("CoverLetterAnalyzer", analyze_cover_letter)
    workflow.add_node("JobMatcher", match_candidates)
    workflow.add_node("FinalRanker", rank_candidates)
    workflow.add_node("supervisor", supervisor_agent)
    
    # Add edges from each agent back to supervisor
    for member in members:
        workflow.add_edge(member, "supervisor")
    
    # Add conditional edges from supervisor
    workflow.add_conditional_edges(
        "supervisor",
        lambda state: state["next"],
        {
            "ResumeAnalyzer": "ResumeAnalyzer",
            "LinkedInAnalyzer": "LinkedInAnalyzer",
            "CoverLetterAnalyzer": "CoverLetterAnalyzer",
            "JobMatcher": "JobMatcher",
            "FinalRanker": "FinalRanker",
            "FINISH": END,
        },
    )
    
    # Set entry point
    workflow.add_edge(START, "supervisor")
    
    # Compile the graph
    graph = workflow.compile()
    
    logger.info("Successfully compiled the HR supervisor graph")
    return graph


# Create and export the graph
graph = create_graph()

