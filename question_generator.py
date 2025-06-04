
import json
import logging
import random
import os
import re
import ollama
from Models.DORA_CHATBOT.EVALUATION.config import *

chunks_file = "document_embeddings/split_documents_deleted.json"  

RANDOM_SEED = 50
NUM_CHUNK_PAIRS = 3000    

# Prompt type selection
PROMPT_TYPE = 'default'

# Default question generation prompt
default_prompt = """
Context: {context}

You are an expert in European financial regulation. Based on the legal text, 
generate one clear question likely to arise in real-world compliance.

Instructions:
- The question must be answerable using only the given context.
- Keep the question specific, concise, and meaningful.

Your task is to generate **exactly one question per chunk**.

Output:
Question {question_num}:
"""

def clean_question(question):
    """Clean and normalize the question text."""
    question = re.sub(r"^.*?Question\s*\d+\s*:\s*", "", question, flags=re.DOTALL)
    question = re.sub(r"^\s*\d+\.|^[-•]+", "", question).strip()
    question = re.sub(r"\s+", " ", question.replace("\u00a0", " ")).strip()
    return question

def setup_question_generator(model_type="ollama", model_name="llama3.2"):
    """Create a function that uses ollama directly instead of langchain"""
    def generate(prompt):
        try:
            response = ollama.chat(
                model="deepseek-coder:6.7b", 
                messages=[{"role": "user", "content": prompt}]
            )
            return response.get("message", {}).get("content", "")
        except Exception as e:
            logging.error(f"Error generating with ollama: {e}")
            return f"Error occurred: {str(e)}"
    
    return generate

def load_chunks_from_file(chunks_file):
    """Load chunks from file."""
    try:
        with open(chunks_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        logging.info(f" Loaded {len(chunks)} chunks from {chunks_file}")
        return chunks
    except Exception as e:
        logging.error(f" Error loading chunks: {e}")
        raise

def generate_question_context_pairs_from_chunks(chunks, llm, num_pairs=NUM_CHUNK_PAIRS, prompt_type='default'):
    output_file = f"question_context_pairs_{prompt_type}.json"
    
    existing_pairs = []
    processed_chunk_numbers = set()
    
    # Check if file exists and load existing pairs
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_pairs = json.load(f)
                print(f" Loading existing question-context pairs from {output_file}")
                logging.info(f"Loaded {len(existing_pairs)} existing question-context pairs from {output_file}")
                
                # Keep track of which chunks have already been processed
                for pair in existing_pairs:
                    processed_chunk_numbers.add(pair["chunk_number"])
                    
                    # Clean questions if needed
                    original_question = pair["question"]
                    cleaned_question = clean_question(original_question)
                    pair["question"] = cleaned_question
                
                logging.info(f"Already processed {len(processed_chunk_numbers)} chunks")
        except json.JSONDecodeError:

            logging.warning(f"{output_file} is corrupted. Starting fresh.")
            existing_pairs = []
            processed_chunk_numbers = set()
    
    # If we already have enough pairs, just return them
    if len(existing_pairs) >= num_pairs:
        logging.info(f"Already have {len(existing_pairs)} pairs, which meets the target of {num_pairs}")
        return existing_pairs
    
    remaining_pairs = num_pairs - len(existing_pairs)
    logging.info(f"Need to generate {remaining_pairs} more questions to reach target of {num_pairs}")
    
    random.seed(RANDOM_SEED)
    # Filter out chunks that have already been processed
    unprocessed_chunks = [chunk for chunk in chunks if chunk["metadata"]["chunk_number"] not in processed_chunk_numbers]
    
    to_process = min(remaining_pairs, len(unprocessed_chunks))
    selected_chunks = random.sample(unprocessed_chunks, to_process)
    
    generation_prompt = default_prompt
    
    # Start index from where we left off
    start_idx = len(existing_pairs)
    
    # Generate new pairs
    new_pairs = []
    for i, chunk in enumerate(selected_chunks):
        context = chunk["text"] if isinstance(chunk, dict) else chunk
        prompt = generation_prompt.format(context=context, question_num=1)
        
        # Log progress
        current_idx = start_idx + i + 1
        logging.info(f"Generating question {current_idx}/{num_pairs} (chunk {chunk['metadata']['chunk_number']})")
        
        question = llm(prompt).strip()  # Use the function directly
        cleaned_question = clean_question(question)
        question_id = f"Q{current_idx:03d}"
        chunk_number = chunk["metadata"]["chunk_number"]
        parent_id = chunk["metadata"].get("parent_id", "unknown")
        
        new_pair = {
            "context": context,
            "question": cleaned_question,
            "chunk_index": start_idx + i,
            "question_id": question_id,
            "chunk_number": chunk_number,
            "parent_id": parent_id
        }
        
        new_pairs.append(new_pair)
        
        # Append to file after each question to save progress
        all_pairs = existing_pairs + new_pairs
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_pairs, f, indent=4, ensure_ascii=False)
        
        logging.info(f"Saved progress: {len(all_pairs)}/{num_pairs} questions")
    
    # Combine existing and new pairs
    final_pairs = existing_pairs + new_pairs
    logging.info(f"Completed! Generated {len(new_pairs)} new questions for a total of {len(final_pairs)}/{num_pairs}")
    
    return final_pairs

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )
    logging.getLogger("urllib3").setLevel(logging.ERROR)
    logging.getLogger("requests").setLevel(logging.ERROR)
    logging.getLogger("openai").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.ERROR)

    llm = setup_question_generator(model_type="ollama", model_name=GENERATOR_MODEL)
    
    chunks = load_chunks_from_file(chunks_file)

    pairs = generate_question_context_pairs_from_chunks(
        chunks=chunks,
        llm=llm,
        num_pairs=NUM_CHUNK_PAIRS,
        prompt_type=PROMPT_TYPE
    )