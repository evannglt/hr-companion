"""LinkedIn profile analysis agent."""

import logging
from typing import Dict

from langchain_core.messages import HumanMessage

from ..state import CandidateState
from ..tools import scrape_linkedin
from ..llm import llm

logger = logging.getLogger(__name__)


def analyze_linkedin(state: CandidateState) -> Dict:
    """Analyze LinkedIn profiles for all candidates.
    
    Extracts and analyzes professional summary, endorsements, connections,
    and career trajectory from LinkedIn profiles.
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state with LinkedIn analyses
    """
    logger.info("LinkedIn Analyzer: Starting LinkedIn profile analysis")
    
    linkedin_analyses = {}
    candidates = state.get("candidates", [])
    
    for idx, candidate in enumerate(candidates):
        candidate_id = candidate.get("id", f"candidate_{idx}")
        linkedin_url = candidate.get("linkedin_url")
        
        if not linkedin_url:
            logger.warning(f"No LinkedIn URL for {candidate_id}")
            linkedin_analyses[candidate_id] = "No LinkedIn profile provided"
            continue
        
        try:
            # Scrape/parse the LinkedIn profile
            linkedin_text = scrape_linkedin(linkedin_url)
            
            if not linkedin_text or linkedin_text.startswith("Error"):
                linkedin_analyses[candidate_id] = linkedin_text or "No content available"
                continue
            
            # Use LLM to analyze the LinkedIn profile
            prompt = f"""Analyze the following LinkedIn profile and extract key insights:

LinkedIn Profile:
{linkedin_text}

Provide a structured analysis including:
1. Professional Summary and headline
2. Career trajectory and progression
3. Key skills and endorsements
4. Notable accomplishments and projects
5. Professional network indicators (if available)
6. Recommendations and testimonials (if available)
7. Overall professional brand and positioning

Be concise but insightful."""

            analysis = llm.invoke(prompt)
            linkedin_analyses[candidate_id] = analysis
            logger.info(f"Completed LinkedIn analysis for {candidate_id}")
            
        except Exception as e:
            logger.error(f"Error analyzing LinkedIn for {candidate_id}: {e}")
            linkedin_analyses[candidate_id] = f"Error analyzing LinkedIn: {str(e)}"
    
    # Create a summary message
    summary = f"LinkedIn Analyzer: Completed analysis of {len(linkedin_analyses)} LinkedIn profiles."
    
    return {
        "messages": [HumanMessage(content=summary, name="LinkedInAnalyzer")],
        "linkedin_analyses": linkedin_analyses,
    }

