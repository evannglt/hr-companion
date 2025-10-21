"""Supervisor agent for coordinating the multi-agent workflow."""

import logging
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel

from .llm import llm
from .state import CandidateState

logger = logging.getLogger(__name__)

# Define the team members
members = ["ResumeAnalyzer", "LinkedInAnalyzer", "CoverLetterAnalyzer", "JobMatcher", "FinalRanker"]
options = ["FINISH"] + members


class RouteResponse(BaseModel):
    """Response model for routing decisions."""
    
    next: Literal[*options]  # type: ignore


# Supervisor system prompt
system_prompt = """You are a supervisor managing a team of specialized agents to evaluate job candidates.

Your team consists of:
- ResumeAnalyzer: Analyzes candidate resumes
- LinkedInAnalyzer: Analyzes LinkedIn profiles
- CoverLetterAnalyzer: Analyzes cover letters
- JobMatcher: Matches candidates against job requirements
- FinalRanker: Creates final ranking and recommendations

WORKFLOW RULES (MANDATORY):
1. If you see NO messages from ResumeAnalyzer → route to ResumeAnalyzer
2. If you see ResumeAnalyzer but NO LinkedInAnalyzer → route to LinkedInAnalyzer
3. If you see LinkedInAnalyzer but NO CoverLetterAnalyzer → route to CoverLetterAnalyzer
4. If you see CoverLetterAnalyzer but NO JobMatcher → route to JobMatcher
5. If you see JobMatcher but NO FinalRanker → route to FinalRanker
6. If you see FinalRanker → route to FINISH

NEVER skip agents. ALWAYS follow this exact order.
Check the message history - each agent adds a message with their name when they complete.
"""


def supervisor_agent(state: CandidateState):
    """Supervisor agent that routes to the next worker.
    
    Args:
        state: Current workflow state
        
    Returns:
        Dict with the next agent to route to
    """
    # Create the prompt with partial binding (following LangGraph docs pattern)
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "Given the conversation above, who should act next? "
            "Or should we FINISH? Select one of: {options}",
        ),
    ]).partial(options=str(options), members=", ".join(members))
    
    # Create supervisor chain (following LangGraph docs pattern)
    supervisor_chain = prompt | llm.with_structured_output(RouteResponse)
    
    # Invoke with state directly
    response = supervisor_chain.invoke(state)
    
    next_agent = response.next
    logger.info(f"Supervisor routing to: {next_agent}")
    
    return {"next": next_agent}

