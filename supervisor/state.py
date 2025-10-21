"""State definition for the HR candidate evaluation workflow."""

import operator
from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import BaseMessage


class CandidateState(TypedDict):
    """State for tracking candidate evaluation progress.
    
    Attributes:
        messages: Message history between supervisor and agents
        job_offer: Text description of the job offer
        candidates: List of candidate information dicts with paths/URLs
        resume_analyses: Resume analysis results per candidate
        linkedin_analyses: LinkedIn analysis results per candidate
        cover_letter_analyses: Cover letter analysis results per candidate
        job_matches: Job matching scores and analysis per candidate
        final_ranking: Final ranking with detailed reasoning
        next: Next agent to route to (set by supervisor)
    """
    
    messages: Annotated[Sequence[BaseMessage], operator.add]
    job_offer: str
    candidates: list[dict]
    resume_analyses: dict
    linkedin_analyses: dict
    cover_letter_analyses: dict
    job_matches: dict
    final_ranking: dict
    next: str

