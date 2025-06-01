
"""
Post Retrieval filtering: Relevance filtering

Uses LLM to score and filter retrieved documents based on relevance to user queries, implementing batch processing.
"""

import json
import logging
import re
from datetime import datetime
from generator import get_azure_llm  
from config import *


def filter_documents_by_llm(retriever, question, retrieved_docs):
    
    """
    Filters retrieved documents using LLM-based relevance scoring.
    
    Args:
        retriever: Document retriever instance
        question (str): User query 
        retrieved_docs (list): Documents to be filtered
        
    Returns:
        list: Documents with relevance score > 3
    """
    if not USE_LLM_FILTERING:
        return retrieved_docs

    if not retrieved_docs:
        return []

    if not hasattr(retriever, 'llm') or retriever.llm is None:
        llm = get_azure_llm()
    else:
        llm = retriever.llm
    
    batch_size = 10
    filtered_docs = []
    
    for batch_idx in range(0, len(retrieved_docs), batch_size):
        batch_end = min(batch_idx + batch_size, len(retrieved_docs))
        batch_docs = retrieved_docs[batch_idx:batch_end]
        
        document_entries = []
        for i, doc in enumerate(batch_docs):
            text = doc.text
            if len(text) > 5000:
                text = text[:5000] + "..."
            document_entries.append(f"Document {i+1}:\n{text}\n")
        

        nl = '\n'
        batch_prompt = f"""
        QUESTION: {question}
        
        ### Relevance Assessment Criteria (Score 0–5):
        Assign a score between 0 and 5 for each document based on how well it helps answer the question.

        - 5 = Very relevant: directly answers the question or provides strong supporting information.
        - 4 = Mostly relevant: discusses the topic and likely contributes to the answer.
        - 3 = Somewhat relevant: may be useful but is incomplete or tangential.
        - 2 = Weakly relevant: touches on the topic but doesn't help much.
        - 1 = Barely relevant: loosely connected, mostly noise.
        - 0 = Not relevant at all: completely unrelated to the question.

        ### Instructions:
        Please assign one score (0–5) per document.

        CONTEXT:
        {nl.join(document_entries)}

        ### Output Format:
        Return a JSON array of scores:
        ```json
        [5, 4, 0, 3, ...]
        ```
        """

        prompt_preview = batch_prompt[:300] + "..." if len(batch_prompt) > 300 else batch_prompt
        logging.info(f"Batch filtering prompt (preview):\n{prompt_preview}")
        
        try:
            response = llm.invoke(batch_prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            logging.info(f"LLM batch filtering response:\n{response_text}")

            match = re.search(r'\[.*?\]', response_text, re.DOTALL)
            if match:
                try:
                    scores = json.loads(match.group(0))
                    logging.info(f" Parsed scores: {scores}")
                    
                    if len(scores) != len(batch_docs):
                        logging.warning(f"Score count mismatch: {len(scores)} for {len(batch_docs)} documents")
                        scores += [5] * (len(batch_docs) - len(scores)) if len(scores) < len(batch_docs) else scores[:len(batch_docs)]
                except json.JSONDecodeError:
                    logging.warning(" Failed to parse JSON. Defaulting all scores to 5.")
                    scores = [5] * len(batch_docs)
            else:
                logging.warning(" No score list found. Using default scores.")
                scores = [5] * len(batch_docs)

            for i, (doc, score) in enumerate(zip(batch_docs, scores)):
                try:
                    score = int(score)
                except:
                    score = 5  # fallback
                if score > 3:
                    filtered_docs.append(doc)
                    logging.info(f"KEPT:-------Doc {batch_idx + i + 1} | Score: {score}")
                else:
                    logging.info(f"DROPPED:----Doc {batch_idx + i + 1} | Score: {score}")

        except Exception as e:
            logging.error(f" Error during batch filtering: {e}")
            filtered_docs.extend(batch_docs)
            for i, doc in enumerate(batch_docs):
                logging.info(f"(Error): Document {batch_idx + i + 1}: {doc.text[:100]}...")

    return filtered_docs