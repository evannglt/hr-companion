"""Cover letter analysis agent."""

import logging
from typing import Dict

from langchain_core.messages import HumanMessage

from ..state import CandidateState
from ..tools import parse_pdf
from ..llm import llm

logger = logging.getLogger(__name__)


def analyze_cover_letter(state: CandidateState) -> Dict:
    """Analyze cover letters for all candidates.
    
    Extracts and analyzes motivation, cultural fit, and communication style
    from candidate cover letters (when provided).
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state with cover letter analyses
    """
    logger.info("Cover Letter Analyzer: Starting cover letter analysis")
    
    cover_letter_analyses = {}
    candidates = state.get("candidates", [])
    
    for idx, candidate in enumerate(candidates):
        candidate_id = candidate.get("id", f"candidate_{idx}")
        cover_letter_path = candidate.get("cover_letter_path")
        
        # Cover letter is optional
        if not cover_letter_path:
            logger.info(f"No cover letter for {candidate_id} (optional)")
            cover_letter_analyses[candidate_id] = "No cover letter provided (optional)"
            continue
        
        try:
            # Parse the cover letter PDF
            cover_letter_text = parse_pdf(cover_letter_path)
            
            if not cover_letter_text.strip():
                cover_letter_analyses[candidate_id] = "Cover letter file is empty"
                continue
            
            # Use LLM to analyze the cover letter
            prompt = f"""Analyze the following cover letter and extract key insights:

Cover Letter:
{cover_letter_text}

Provide a structured analysis including:
1. Motivation for applying
2. Understanding of the role and company
3. Cultural fit indicators
4. Communication style and professionalism
5. Unique value propositions mentioned
6. Passion and enthusiasm level
7. Writing quality and clarity

Be concise but insightful."""

            analysis = llm.invoke(prompt)
            cover_letter_analyses[candidate_id] = analysis
            logger.info(f"Completed cover letter analysis for {candidate_id}")
            
        except FileNotFoundError:
            logger.warning(f"Cover letter file not found for {candidate_id}")
            cover_letter_analyses[candidate_id] = "Cover letter file not found"
        except Exception as e:
            logger.error(f"Error analyzing cover letter for {candidate_id}: {e}")
            cover_letter_analyses[candidate_id] = f"Error analyzing cover letter: {str(e)}"
    
    # Create a summary message
    summary = f"Cover Letter Analyzer: Completed analysis of {len(cover_letter_analyses)} cover letters."
    
    return {
        "messages": [HumanMessage(content=summary, name="CoverLetterAnalyzer")],
        "cover_letter_analyses": cover_letter_analyses,
    }

