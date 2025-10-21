"""Final ranking agent."""

import json
import logging
from typing import Dict

from langchain_core.messages import HumanMessage

from ..state import CandidateState
from ..llm import llm

logger = logging.getLogger(__name__)


def rank_candidates(state: CandidateState) -> Dict:
    """Produce final ranking of candidates.
    
    Aggregates all analyses to create a final ranking with detailed
    reasoning, generating both a detailed report and structured JSON output.
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state with final ranking
    """
    logger.info("Final Ranker: Starting final ranking process")
    
    job_offer = state.get("job_offer", "")
    candidates = state.get("candidates", [])
    job_matches = state.get("job_matches", {})
    resume_analyses = state.get("resume_analyses", {})
    linkedin_analyses = state.get("linkedin_analyses", {})
    cover_letter_analyses = state.get("cover_letter_analyses", {})
    
    try:
        # Compile all information for ranking
        candidates_summary = []
        for idx, candidate in enumerate(candidates):
            candidate_id = candidate.get("id", f"candidate_{idx}")
            candidates_summary.append(f"""
CANDIDATE: {candidate_id}
Resume: {resume_analyses.get(candidate_id, 'N/A')}
LinkedIn: {linkedin_analyses.get(candidate_id, 'N/A')}
Cover Letter: {cover_letter_analyses.get(candidate_id, 'N/A')}
Job Match: {job_matches.get(candidate_id, 'N/A')}
---
""")
        
        all_candidates_text = "\n".join(candidates_summary)
        
        # Use LLM to create final ranking
        prompt = f"""Based on all the analyses, create a final ranking of candidates for the job offer.

JOB OFFER:
{job_offer}

CANDIDATE ANALYSES:
{all_candidates_text}

Provide:
1. A ranked list of candidates (best to least suitable)
2. For each candidate, provide:
   - Overall score (0-100)
   - Key strengths
   - Key weaknesses
   - Reasoning for their ranking
   - Hiring recommendation (Strong Yes / Yes / Maybe / No)
3. A summary comparing the top candidates
4. Final recommendation

Be thorough, objective, and provide actionable insights for the hiring decision."""

        detailed_report = llm.invoke(prompt)
        
        # Create structured JSON output
        json_prompt = f"""Based on the following detailed ranking report, extract a structured JSON ranking.

REPORT:
{detailed_report}

Return ONLY a valid JSON object with this structure:
{{
    "rankings": [
        {{
            "candidate_id": "candidate_id",
            "rank": 1,
            "score": 85,
            "recommendation": "Strong Yes",
            "key_strengths": ["strength1", "strength2"],
            "key_weaknesses": ["weakness1", "weakness2"]
        }}
    ],
    "top_recommendation": "candidate_id",
    "summary": "Brief summary of the decision"
}}

Ensure the JSON is valid and complete."""

        json_response = llm.invoke(json_prompt)
        
        # Try to parse the JSON (with error handling)
        try:
            # Extract JSON from response if it's wrapped in markdown or text
            json_str = str(json_response)
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0].strip()
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0].strip()
            
            structured_output = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON from LLM response: {e}")
            # Create a fallback structure
            structured_output = {
                "rankings": [
                    {
                        "candidate_id": candidate.get("id", f"candidate_{idx}"),
                        "rank": idx + 1,
                        "score": 0,
                        "recommendation": "Unable to parse",
                        "key_strengths": [],
                        "key_weaknesses": []
                    }
                    for idx, candidate in enumerate(candidates)
                ],
                "top_recommendation": candidates[0].get("id", "candidate_0") if candidates else None,
                "summary": "Error parsing structured output"
            }
        
        final_ranking = {
            "detailed_report": str(detailed_report),
            "structured_output": structured_output,
        }
        
        logger.info("Completed final ranking")
        
        # Create a summary message
        summary = f"Final Ranker: Completed ranking of {len(candidates)} candidates. Top recommendation: {structured_output.get('top_recommendation', 'N/A')}"
        
        return {
            "messages": [HumanMessage(content=summary, name="FinalRanker")],
            "final_ranking": final_ranking,
        }
        
    except Exception as e:
        logger.error(f"Error in final ranking: {e}")
        error_ranking = {
            "detailed_report": f"Error generating ranking: {str(e)}",
            "structured_output": {"error": str(e)},
        }
        
        return {
            "messages": [HumanMessage(content=f"Final Ranker: Error - {str(e)}", name="FinalRanker")],
            "final_ranking": error_ranking,
        }

