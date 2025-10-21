"""LinkedIn scraping tool using bs4 and LangChain."""

import logging
from urllib.parse import urlparse

import bs4
from langchain_community.document_loaders import WebBaseLoader

logger = logging.getLogger(__name__)


def scrape_linkedin(url_or_text: str) -> str:
    """Extract LinkedIn profile information.
    
    This function supports two modes:
    1. URL scraping: If a valid URL is provided, scrapes the LinkedIn page
    2. Text mode: If plain text is provided, returns it as-is
    
    Args:
        url_or_text: Either a LinkedIn URL to scrape or plain text content
        
    Returns:
        Extracted LinkedIn profile text
    """
    try:
        # Check if it's a URL
        parsed = urlparse(url_or_text)
        is_url = bool(parsed.scheme and parsed.netloc)
        
        if not is_url:
            # Treat as plain text
            logger.info("Treating input as plain text LinkedIn profile")
            return url_or_text
        
        # Scrape the URL
        logger.info(f"Scraping LinkedIn URL: {url_or_text}")
        
        # Use WebBaseLoader with bs4 to extract relevant content
        loader = WebBaseLoader(
            web_paths=(url_or_text,),
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(
                    class_=(
                        "profile",
                        "profile-section",
                        "pv-top-card",
                        "experience-section",
                        "education-section",
                        "skill-section",
                    )
                )
            ),
        )
        
        docs = loader.load()
        
        if not docs:
            logger.warning(f"No content extracted from URL: {url_or_text}")
            return ""
        
        # Combine all document content
        text = "\n\n".join(doc.page_content for doc in docs)
        
        logger.info(f"Successfully scraped {len(text)} characters from LinkedIn")
        return text
        
    except Exception as e:
        logger.error(f"Error scraping LinkedIn {url_or_text}: {e}")
        # Return empty string on error rather than failing
        return f"Error accessing LinkedIn profile: {str(e)}"

