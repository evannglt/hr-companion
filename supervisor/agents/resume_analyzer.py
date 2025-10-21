"""Resume analysis agent."""

import logging
from typing import Dict

from langchain_core.messages import HumanMessage

from ..state import CandidateState
from ..tools import parse_pdf
from ..llm import llm

logger = logging.getLogger(__name__)


def analyze_resume(state: CandidateState) -> Dict:
    """Analyze resumes for all candidates.
    
    Extracts and analyzes skills, experience, education, and qualifications
    from candidate resume PDFs.
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state with resume analyses
    """
    logger.info("Resume Analyzer: Starting resume analysis")
    
    resume_analyses = {}
    candidates = state.get("candidates", [])
    
    for idx, candidate in enumerate(candidates):
        candidate_id = candidate.get("id", f"candidate_{idx}")
        resume_path = candidate.get("resume_path")
        
        if not resume_path:
            logger.warning(f"No resume path for {candidate_id}")
            resume_analyses[candidate_id] = "No resume provided"
            continue
        
        try:
            # Parse the resume PDF
            resume_text = parse_pdf(resume_path)
            
            # Use LLM to analyze the resume
            prompt = f"""Analyze the following resume and extract key information:

Resume Content:
{resume_text}

Provide a structured analysis including:
1. Skills (technical and soft skills)
2. Work Experience (positions, companies, duration, key achievements)
3. Education (degrees, institutions, graduation dates)
4. Certifications and additional qualifications
5. Overall strengths and areas of expertise

Be concise but thorough."""

            analysis = llm.invoke(prompt)
            resume_analyses[candidate_id] = analysis
            logger.info(f"Completed resume analysis for {candidate_id}")
            
        except Exception as e:
            logger.error(f"Error analyzing resume for {candidate_id}: {e}")
            resume_analyses[candidate_id] = f"Error analyzing resume: {str(e)}"
    
    # Create a summary message
    summary = f"Resume Analyzer: Completed analysis of {len(resume_analyses)} candidate resumes."
    
    return {
        "messages": [HumanMessage(content=summary, name="ResumeAnalyzer")],
        "resume_analyses": resume_analyses,
    }

