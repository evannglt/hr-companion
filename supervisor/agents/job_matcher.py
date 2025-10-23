"""Job matching agent."""

import logging
from typing import Dict

from langchain_core.messages import HumanMessage

from ..state import CandidateState
from ..llm import llm

logger = logging.getLogger(__name__)


def match_candidates(state: CandidateState) -> Dict:
    """Match candidates against job offer requirements.
    
    Compares candidate profiles against the job offer to identify
    compatibility, strengths, and potential gaps.
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state with job matching results
    """
    logger.info("Job Matcher: Starting candidate-job matching")
    
    job_offer = state.get("job_offer", "")
    candidates = state.get("candidates", [])
    resume_analyses = state.get("resume_analyses", {})
    linkedin_analyses = state.get("linkedin_analyses", {})
    cover_letter_analyses = state.get("cover_letter_analyses", {})
    
    job_matches = {}
    
    for idx, candidate in enumerate(candidates):
        candidate_id = candidate.get("id", f"candidate_{idx}")
        
        # Gather all analyses for this candidate
        resume_analysis = resume_analyses.get(candidate_id, "No resume analysis")
        linkedin_analysis = linkedin_analyses.get(candidate_id, "No LinkedIn analysis")
        cover_letter_analysis = cover_letter_analyses.get(candidate_id, "No cover letter analysis")
        
        try:
            # Use LLM to match candidate against job requirements
            prompt = f"""Match the following candidate against the job offer requirements:

JOB OFFER:
{job_offer}

CANDIDATE PROFILE:

Resume Analysis:
{resume_analysis}

LinkedIn Analysis:
{linkedin_analysis}

Cover Letter Analysis:
{cover_letter_analysis}

Provide a comprehensive matching assessment including:
1. Overall compatibility score (0-100)
2. Key strengths that align with the job requirements
3. Gaps or areas of concern
4. Technical skills match
5. Experience level match
6. Cultural fit indicators
7. Unique advantages this candidate brings
8. Potential risks or limitations

Be objective and thorough in your assessment."""

            match_result_response = llm.invoke(prompt)
            job_matches[candidate_id] = match_result_response.content
            logger.info(f"Completed job matching for {candidate_id}")
            
        except Exception as e:
            logger.error(f"Error matching candidate {candidate_id}: {e}")
            job_matches[candidate_id] = f"Error in job matching: {str(e)}"
    
    # Create a summary message
    summary = f"Job Matcher: Completed matching analysis for {len(job_matches)} candidates against the job offer."
    
    return {
        "messages": [HumanMessage(content=summary, name="JobMatcher")],
        "job_matches": job_matches,
    }

