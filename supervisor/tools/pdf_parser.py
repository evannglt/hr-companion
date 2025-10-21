"""PDF parsing tool using PyPDF2."""

import logging
from pathlib import Path

from PyPDF2 import PdfReader

logger = logging.getLogger(__name__)


def parse_pdf(file_path: str) -> str:
    """Extract text from a PDF file.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Extracted text content from the PDF
        
    Raises:
        FileNotFoundError: If the PDF file doesn't exist
        Exception: For other PDF parsing errors
    """
    try:
        path = Path(file_path)
        if not path.exists():
            logger.error(f"PDF file not found: {file_path}")
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        reader = PdfReader(str(path))
        text_content = []
        
        for page_num, page in enumerate(reader.pages, 1):
            try:
                text = page.extract_text()
                if text.strip():
                    text_content.append(text)
            except Exception as e:
                logger.warning(f"Failed to extract text from page {page_num}: {e}")
                continue
        
        full_text = "\n\n".join(text_content)
        
        if not full_text.strip():
            logger.warning(f"No text extracted from PDF: {file_path}")
            return ""
        
        logger.info(f"Successfully extracted {len(full_text)} characters from {file_path}")
        return full_text
        
    except FileNotFoundError:
        raise
    except Exception as e:
        logger.error(f"Error parsing PDF {file_path}: {e}")
        raise Exception(f"Error parsing PDF: {e}")

