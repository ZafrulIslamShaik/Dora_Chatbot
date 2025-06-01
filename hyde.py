"""
HyDE (Hypothetical Document Embeddings) Module

"""

from llama_index.core.prompts import PromptTemplate
from datetime import datetime

def get_dora_hyde_answer(llm, question: str) -> str:
    """
    Generate hypothetical expert answer for DORA-related queries using HyDE technique.

    Args:
        llm: Language model instance for answer generation
        question (str): Original user query about DORA regulations
        
    Returns:
        str: Generated hypothetical expert answer for embedding
    """
    hyde_prompt = "You are an expert in the Digital Operational Resilience Act Europe (DORA) and related regulations.Please write a passage to answer the question: {query_str}"
    
    start_time = datetime.now()
    
    template = PromptTemplate(hyde_prompt)
    prompt = template.format(query_str=question)
    
    response = llm.invoke(prompt)
    end_time = datetime.now()
    
    hyde_answer = response.content if hasattr(response, 'content') else response
    
    return hyde_answer