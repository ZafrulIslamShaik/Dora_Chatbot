import json
import logging
from typing import List, Dict, Any, Optional
import os
from generator import get_azure_llm
from Models.DORA_CHATBOT.EVALUATION.config import *

# Configuration
CHUNK_TYPE = "filtered_chunks"    # or   Normal_retrieved_chunks
RET_RESULTS_FILE = "ret_results.json"
SPLIT_DOCUMENTS_FILE = "split_documents.json"  
START_INDEX = 301
END_INDEX = 400
BATCH_NUMBER = 1  

def setup_logging():
    """Configure logging for the application"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("rag_generator.log")
        ]
    )
    return logging.getLogger(__name__)

def load_json_file(file_path: str) -> Any:
    """Load and parse a JSON file"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading file {file_path}: {e}")
        raise

def load_retrieval_results(file_path: str) -> List[Dict[str, Any]]:
    """Load retrieval results from JSON file"""
    logging.info(f"Loading retrieval results from {file_path}")
    return load_json_file(file_path)

def load_split_documents(file_path: str) -> List[Dict[str, Any]]:
    """Load split documents from JSON file"""
    logging.info(f"Loading split documents from {file_path}")
    return load_json_file(file_path)

def build_document_index(split_documents: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Build an index of documents by parent_id AND chunk_number for quick lookup"""
    document_index = {}
    for doc in split_documents:
        parent_id = doc.get("metadata", {}).get("parent_id")
        chunk_number = doc.get("metadata", {}).get("chunk_number")
        if parent_id and chunk_number:
            key = f"{parent_id}_{chunk_number}"
            document_index[key] = doc
    logging.info(f"Built document index with {len(document_index)} unique chunks")
    return document_index

def get_document_content(parent_id: str, chunk_number: int, document_index: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Get document content for a specific parent_id and chunk_number"""
    key = f"{parent_id}_{chunk_number}"
    return document_index.get(key, {})

def format_chunk_for_context(doc: Dict[str, Any]) -> str:
    """Format a document chunk for inclusion in the context"""
    if not doc:
        return "ERROR: Document not found"
    
    metadata = doc.get("metadata", {})
    content = doc.get("text", "")
    chunk_number = metadata.get("chunk_number", "")
    
    return f"Chunk {chunk_number}: {content}"

def get_text_preview(text: str, max_length: int = 20000) -> str:
    """Get a preview of the text, limited to max_length characters"""
    if not text:
        return ""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."

def generate_answers(
    retrieval_results: List[Dict[str, Any]], 
    document_index: Dict[str, Dict[str, Any]],
    llm,
    chunk_type: str
) -> List[Dict[str, Any]]:
    """Generate answers for each question using the specified chunk type"""
    import time
    results = []
    
    start_num = START_INDEX
    end_num = END_INDEX
    
    logging.info(f"Processing questions from Q{start_num:03d} to Q{end_num:03d} (Batch {BATCH_NUMBER})")
    
    questions_to_process = []
    for item in retrieval_results:
        question_id = item.get("question_id", "")
        
        str_question_id = str(question_id) if question_id else ""
        
        # Skip if question ID doesn't match our pattern
        try:
            if str_question_id.startswith('Q'):
                q_num = int(str_question_id[1:])
                if q_num < start_num or q_num > end_num:
                    continue 
            else:
                continue  
        except ValueError:
            continue 
            
        # Add to questions to process
        questions_to_process.append(item)
    
    # Process filtered questions
    for i, item in enumerate(questions_to_process):
        try:
            question_id = item.get("question_id", "")
            logging.info(f"Processing question {i+1}/{len(questions_to_process)}: {question_id} - {item.get('question', '')[:50]}...")
            
            # Add a delay before processing each question
            if i > 0:
                logging.info(f"Waiting 5 seconds before processing the next question...")
                time.sleep(30) 
            
            question = item["question"]
            
            # Get chunks from the specified chunk type
            chunks_with_metadata = item.get(chunk_type, [])
            if not chunks_with_metadata:
                logging.warning(f"No chunks found for chunk type '{chunk_type}' in question {question_id}")
                continue
            
            # Get content for each specific chunk using both parent_id and chunk_number
            chunks_content = []
            chunk_texts_preview = []  
            
            for chunk in chunks_with_metadata:
                parent_id = chunk.get("metadata", {}).get("parent_id")
                chunk_number = chunk.get("metadata", {}).get("chunk_number")
                
                if parent_id and chunk_number:
                    doc = get_document_content(parent_id, chunk_number, document_index)
                    
                    if doc:
                        chunks_content.append(format_chunk_for_context(doc))
                        text_preview = get_text_preview(doc.get("text", ""))
                        chunk_texts_preview.append({
                            "parent_id": parent_id,
                            "chunk_number": chunk_number,
                            "text_preview": text_preview
                        })
                    else:
                        logging.warning(f"Document not found for parent_id={parent_id}, chunk_number={chunk_number}")
            
            # Combine all chunks into a single context
            context = "\n\n" + "="*50 + "\n\n".join(chunks_content)
            
            # Generate answer using LLM
            prompt = f"""
            You are a Digital Operational Resilience Act (DORA) expert assistant.
            
            INSTRUCTIONS:
            1. Answer ONLY using information from the context chunks provided above.
            2. If the answer cannot be determined from the chunks, respond: "I don't know. The answer is not in the provided documents."
            3. DO NOT use any prior knowledge beyond what's necessary to interpret the chunks.
            4. After your answer, you MUST include a "CHUNKS USED:" section formatted exactly as shown:
             
            ### **Format Example:**
          
            #### **Answer:**
            XXXXXXX
        

            ### **Now, apply the same format to the question below:**

            **QUESTION:**  
            {question}  

            **CONTEXT:**  
            {context}  

            ### **ANSWER:**
            """
            
            logging.info(f"Generating answer for question {question_id}...")
            try:
                response = llm.invoke(prompt)
                answer = response.content
                logging.info(f"Successfully received response from LLM for question {question_id}")
            except Exception as llm_error:
                logging.error(f"Error getting LLM response for question {question_id}: {llm_error}")
                # Add longer delay if we hit a rate limit
                logging.info("Adding additional delay due to LLM error...")
                time.sleep(5)  # Longer delay on error
                # Try once more
                try:
                    response = llm.invoke(prompt)
                    answer = response.content
                    logging.info(f"Successfully received response from LLM on retry for question {question_id}")
                except Exception as retry_error:
                    logging.error(f"Error on retry for question {question_id}: {retry_error}")
                    answer = f"ERROR: Failed to generate answer due to: {str(retry_error)}"
            
            # Store the result
            results.append({
                "question_id": question_id,
                "batch_number": BATCH_NUMBER,               
                "question": question,
                "chunk_type": chunk_type,
                "chunk_texts": chunk_texts_preview,  
                "answer": answer
            })
            
            logging.info(f"Successfully generated answer for question {question_id}")
            
        except Exception as e:
            logging.error(f"Error generating answer for question {i+1}: {e}")
    
    return results

def save_results(results: List[Dict[str, Any]], file_path: str) -> None:
    """Save results to a JSON file"""
    try:
        existing_results = []
        
        # Try to load existing results
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    existing_results = json.load(f)
                logging.info(f"Loaded {len(existing_results)} existing results from {file_path}")
            except Exception as e:
                logging.error(f"Error loading existing results from {file_path}: {e}")
                existing_results = []
        
        combined_results = existing_results + results
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(combined_results, f, indent=4, ensure_ascii=False)
        logging.info(f"Results saved successfully to {file_path}")
    except Exception as e:
        logging.error(f"Error saving results to {file_path}: {e}")

def main():
    logger = setup_logging()
    logger.info("Starting RAG answer generator")
    
    try:
        retrieval_results = load_retrieval_results(RET_RESULTS_FILE)
        if not retrieval_results:
            logger.error(f"No retrieval results found in {RET_RESULTS_FILE}")
            return
        
        split_documents = load_split_documents(SPLIT_DOCUMENTS_FILE)
        if not split_documents:
            logger.error(f"No split documents found in {SPLIT_DOCUMENTS_FILE}")
            return
        
        document_index = build_document_index(split_documents)
        

        llm = get_azure_llm()
        
        # Generate answers
        results = generate_answers(retrieval_results, document_index, llm, CHUNK_TYPE)
        
        # Save results
        results_file = f"gen_answers.json"
        save_results(results, results_file)
        
        logger.info(f"Process completed. Results saved to {results_file}")
    
    except Exception as e:
        logger.error(f"An error occurred in the main process: {e}")

if __name__ == "__main__":
    main()