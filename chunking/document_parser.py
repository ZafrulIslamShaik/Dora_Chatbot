"""
This module processes legal documents from web sources, extracting structured content 
from HTML format.

Input: documents_urls.json - Contains URLs and metadata for legal documents
Output: parsed_documents.json - Structured legal content ready for processing
"""

import requests
from bs4 import BeautifulSoup
import json
import logging

# Configure logging for process monitoring and debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_article(article_div, document_name, current_chapter):
    """
    Extracts structured content from individual legal articles or regulations.
    
    Processes HTML div elements containing legal articles/regulations and extracts:
    - Article/regulation identification numbers
    - Titles and subtitles
    - Full text content with hyperlink processing
    - Reference links and metadata
    - Chapter context information
    
    Args:
        article_div (BeautifulSoup element): HTML div containing the article/regulation
        document_name (str): Name of the source legal document
        current_chapter (str): Current chapter context (e.g., "CHAPTER I")
        
    Returns:
        dict: Structured document containing text and metadata, or None if invalid
    """
    article_id = article_div.get("id")
    if not article_id or not (article_id.startswith("art_") or article_id.startswith("rct_")):
        logger.warning(f"Invalid article ID: {article_id}")
        return None

    # Build comprehensive metadata structure for document hierarchy
    metadata = {
        "document": document_name,
        "id": article_id,
        "chapter": current_chapter  
    }
    if article_id.startswith("art_"):
        metadata["article number"] = article_id.replace("art_", "")
    else:
        metadata["regulation number"] = article_id.replace("rct_", "")

    # Extract article titles and subtitles from structured HTML elements
    article_title_tag = article_div.find("p", class_="oj-ti-art")
    article_name = article_title_tag.get_text(strip=True) if article_title_tag else ""

    subtitle_tag = article_div.find("p", class_="oj-sti-art")
    subtitle = subtitle_tag.get_text(strip=True) if subtitle_tag else ""

    metadata["title"] = subtitle or article_name or f"Regulation {article_id}"

    # Extract and preserve reference links for legal cross-referencing
    references = []
    for ref_tag in article_div.find_all("a", class_="oj-ref"):
        ref_text = ref_tag.get_text(strip=True)
        ref_url = ref_tag.get("href")
        if ref_text and ref_url:
            references.append({"text": ref_text, "url": ref_url})

    if references:
        metadata["References"] = references
        logger.info(f"{len(metadata['References'])} reference(s) found")

    # Process article content while handling hyperlinks appropriately
    full_text = []
    seen_texts = set()

    for element in article_div.find_all(["p", "td"]):
        if not any(cls in element.get("class", []) for cls in ["oj-ti-art", "oj-sti-art"]):
            # Create working copy to avoid modifying original HTML structure
            element_copy = BeautifulSoup(str(element), "html.parser")
            
            # Differential link processing: remove footnotes
            for link in element_copy.find_all("a"):
                if (link.get("id", "").startswith("ntc") or 
                    link.get("href", "").startswith("#ntr")):
                    link.decompose()
                else:
                    # Remove content links completely
                    link.decompose()
            
            # Extract and normalize text content
            text = clean_text(element_copy.get_text())
            
            # Prevent duplicate content inclusion
            if text and text not in seen_texts:
                full_text.append(text)
                seen_texts.add(text)

    if not full_text:
        return None
    
    # Structure final document for downstream processing
    document = {
        "text": " ".join(full_text),
        "metadata": metadata
    }

    return document


def clean_text(text):
    """
    Standardizes text formatting by removing non-breaking spaces and normalizing whitespace.
    
    Args:
        text (str): Raw text content extracted from HTML
        
    Returns:
        str: Cleaned and normalized text
    """
    replacements = {'\u00a0': ' '}
    for old, new in replacements.items():
        text = text.replace(old, new)
    return ' '.join(text.split())


def main(start_index=0, count=1):
    try:
        logger.info("Loading document URLs...")
        
        # Load document configuration from input file
        with open("documents_urls.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            document_list = data.get("documents", [])

        # Support batch processing for large document collections
        selected_documents = document_list[start_index:start_index + count]
        if not selected_documents:
            logger.error("No documents found for the given index range.")
            return
        
        all_documents = []
        
        # Process each document URL in the selected batch
        for doc in selected_documents:
            url, document_name = doc["url"], doc["document_name"]
            logger.info(f"Processing: {document_name}")

            # Fetch and parse HTML content with proper encoding handling
            response = requests.get(url)
            response.encoding = 'utf-8'
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            chunk_number = 1
            current_chapter = None  

            # Sequential processing to maintain document structure and hierarchy
            for div in soup.find_all("div"):
                # Track chapter progression for proper document organization
                if div.get("id", "").startswith("cpt_"):
                    chapter_tag = div.find("p", class_="oj-ti-section-1")
                    if chapter_tag:
                        italic_tag = chapter_tag.find("span", class_="oj-italic")
                        if italic_tag:
                            current_chapter = italic_tag.get_text(strip=True)  
                            logger.info(f"Found {current_chapter}")

                # Extract individual articles and regulations within chapter context
                if div.get("id", "").startswith(("art_", "rct_")):
                    document = extract_article(div, document_name, current_chapter)
                    if document:
                        document["metadata"]["chunk_number"] = chunk_number
                        chunk_number += 1
                        all_documents.append(document)

        # Save structured output for downstream processing pipeline
        if all_documents:
            output_filename = "parsed_documents.json"
            with open(output_filename, 'w', encoding='utf-8') as f:
                json.dump(all_documents, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Successfully saved {len(all_documents)} document(s) to {output_filename}")

        logger.info("Process completed successfully")

    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    # Process documents 0-32 from the input configuration
    main(start_index=0, count=33)