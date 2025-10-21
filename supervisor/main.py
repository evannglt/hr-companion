"""Entry point for the HR candidate evaluation system."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from langchain_core.messages import HumanMessage

from .graph import graph

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def evaluate_candidates(
    job_offer: str,
    candidates: List[Dict[str, str]],
    verbose: bool = True
) -> Dict:
    """Evaluate candidates for a job offer.
    
    This is the main entry point for the HR candidate evaluation system.
    It coordinates multiple specialized agents to analyze candidates and
    produce a final ranking with recommendations.
    
    Args:
        job_offer: Text description of the job offer/requirements
        candidates: List of candidate info dicts, each containing:
            - id: Unique identifier for the candidate
            - resume_path: Path to PDF resume
            - linkedin_url: LinkedIn profile URL or text
            - cover_letter_path: (Optional) Path to PDF cover letter
        verbose: Whether to print progress messages
        
    Returns:
        Dictionary containing:
            - detailed_report: Comprehensive text report
            - structured_output: JSON-formatted ranking and scores
            - all_analyses: Complete analysis data for all candidates
            
    Example:
        >>> candidates = [
        ...     {
        ...         "id": "john_doe",
        ...         "resume_path": "resumes/john_doe.pdf",
        ...         "linkedin_url": "https://linkedin.com/in/johndoe",
        ...         "cover_letter_path": "cover_letters/john_doe.pdf"
        ...     },
        ...     {
        ...         "id": "jane_smith",
        ...         "resume_path": "resumes/jane_smith.pdf",
        ...         "linkedin_url": "Jane Smith is a senior developer...",
        ...     }
        ... ]
        >>> job_offer = "Senior Python Developer with 5+ years experience..."
        >>> result = evaluate_candidates(job_offer, candidates)
        >>> print(result['detailed_report'])
        >>> print(json.dumps(result['structured_output'], indent=2))
    """
    logger.info(f"Starting evaluation for {len(candidates)} candidates")
    
    # Create initial message to start the workflow
    initial_message = HumanMessage(
        content=f"Please evaluate {len(candidates)} candidates for the following job offer. "
        f"Follow the complete workflow: analyze resumes, LinkedIn profiles, cover letters, "
        f"match candidates to the job, and provide final rankings.",
        name="user"
    )
    
    # Initialize state
    initial_state = {
        "messages": [initial_message],
        "job_offer": job_offer,
        "candidates": candidates,
        "resume_analyses": {},
        "linkedin_analyses": {},
        "cover_letter_analyses": {},
        "job_matches": {},
        "final_ranking": {},
        "next": "",
    }
    
    # Run the graph
    if verbose:
        print(f"\n{'='*60}")
        print(f"HR CANDIDATE EVALUATION SYSTEM")
        print(f"{'='*60}")
        print(f"\nEvaluating {len(candidates)} candidates...")
        print(f"Job Offer: {job_offer[:100]}..." if len(job_offer) > 100 else f"Job Offer: {job_offer}")
        print(f"\nStarting multi-agent analysis...\n")
    
    try:
        final_state = graph.invoke(initial_state)
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"EVALUATION COMPLETE")
            print(f"{'='*60}\n")
        
        # Extract results
        final_ranking = final_state.get("final_ranking", {})
        detailed_report = final_ranking.get("detailed_report", "No report generated")
        structured_output = final_ranking.get("structured_output", {})
        
        result = {
            "detailed_report": detailed_report,
            "structured_output": structured_output,
            "all_analyses": {
                "resume_analyses": final_state.get("resume_analyses", {}),
                "linkedin_analyses": final_state.get("linkedin_analyses", {}),
                "cover_letter_analyses": final_state.get("cover_letter_analyses", {}),
                "job_matches": final_state.get("job_matches", {}),
            }
        }
        
        logger.info("Evaluation completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}", exc_info=True)
        raise


def main():
    """Example usage of the HR candidate evaluation system."""
    
    # Example job offer
    job_offer = """
    Senior Python Developer
    
    We are seeking an experienced Python developer to join our growing team.
    
    Requirements:
    - 5+ years of professional Python development experience
    - Strong experience with web frameworks (Django, Flask, FastAPI)
    - Experience with cloud platforms (AWS, GCP, or Azure)
    - Proficiency in SQL and NoSQL databases
    - Experience with containerization (Docker, Kubernetes)
    - Strong understanding of software design patterns and best practices
    - Excellent communication and teamwork skills
    
    Nice to have:
    - Experience with machine learning frameworks
    - DevOps experience (CI/CD, infrastructure as code)
    - Open source contributions
    - Bachelor's degree in Computer Science or related field
    
    We offer competitive salary, remote work options, and great benefits.
    """
    
    # Example candidates
    # NOTE: In real usage, replace these with actual file paths and URLs
    candidates = [
        {
            "id": "candidate_1",
            "resume_path": "path/to/candidate1_resume.pdf",
            "linkedin_url": """John Doe - Senior Python Developer
            
            Professional with 6 years of Python development experience.
            Strong background in Django and FastAPI.
            Worked with AWS and containerization.
            Previous roles at tech startups.
            Skills: Python, Django, FastAPI, AWS, Docker, PostgreSQL
            """,
            "cover_letter_path": "path/to/candidate1_cover_letter.pdf",
        },
        {
            "id": "candidate_2",
            "resume_path": "path/to/candidate2_resume.pdf",
            "linkedin_url": """Jane Smith - Full Stack Developer
            
            4 years of full-stack development experience.
            Experience with Python, JavaScript, and React.
            Background in both frontend and backend development.
            Skills: Python, Flask, React, MongoDB, Docker
            """,
        },
    ]
    
    print("\n" + "="*60)
    print("HR CANDIDATE EVALUATION - EXAMPLE RUN")
    print("="*60)
    print("\nNOTE: This is an example with placeholder file paths.")
    print("In real usage, provide actual PDF file paths and LinkedIn URLs.")
    print("="*60 + "\n")
    
    try:
        # Run evaluation
        result = evaluate_candidates(job_offer, candidates, verbose=True)
        
        # Display results
        print("\n" + "="*60)
        print("DETAILED REPORT")
        print("="*60 + "\n")
        print(result["detailed_report"])
        
        print("\n" + "="*60)
        print("STRUCTURED OUTPUT (JSON)")
        print("="*60 + "\n")
        print(json.dumps(result["structured_output"], indent=2))
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nThis example uses placeholder paths. To use the system:")
        print("1. Prepare your candidate PDF files (resumes and cover letters)")
        print("2. Gather LinkedIn URLs or profile text")
        print("3. Call evaluate_candidates() with actual file paths")


if __name__ == "__main__":
    main()

