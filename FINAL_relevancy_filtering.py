import json
import re
import os
from datetime import datetime
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
import time

load_dotenv()

START_INDEX = 306
END_INDEX = 400

BATCH_SIZE = 1
LLM_BATCH_SIZE = 30

RETRIEVAL_RESULTS_FILE = "ret_results_1_500_Dense_10.json"
SPLIT_DOCUMENTS_FILE = "split_documents_with_references.json"
FILTERED_OUTPUT_FILE = "ret_results_filtered_Dense_10.json"

def get_azure_llm():
    try:
        from generator import get_azure_llm as original_get_azure_llm
        return original_get_azure_llm()
    except ImportError:
        from openai import AzureOpenAI
        client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        return client

def load_json_file(file_path: str, default_value=None) -> Any:
    if not os.path.exists(file_path):
        return default_value
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return default_value

def save_json_file(data: Any, file_path: str) -> None:
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        print(f"Successfully saved data to {file_path}")
    except Exception as e:
        print(f"Error saving to {file_path}: {e}")
        raise

def get_document_text(split_documents: List[Dict], parent_id: str, chunk_number: int) -> str:
    for doc in split_documents:
        if (doc.get("metadata", {}).get("parent_id") == parent_id and 
            doc.get("metadata", {}).get("chunk_number") == chunk_number):
            text = doc.get("text", "")
            if isinstance(text, list):
                return " ".join(text)
            return text
    return f"[Document with parent_id {parent_id} and chunk_number {chunk_number} not found]"

def filter_documents_by_relevance(llm, question: str, chunks: List[Dict], split_documents: List[Dict], llm_batch_size: int) -> List[Dict]:
    if not chunks:
        return []
    
    filter_start = datetime.now()
    filtered_chunks = []
    
    for batch_idx in range(0, len(chunks), llm_batch_size):
        batch_end = min(batch_idx + llm_batch_size, len(chunks))
        batch_chunks = chunks[batch_idx:batch_end]
        
        document_entries = []
        for i, chunk in enumerate(batch_chunks):
            parent_id = chunk.get("metadata", {}).get("parent_id")
            chunk_number = chunk.get("metadata", {}).get("chunk_number")
            if parent_id and chunk_number:
                text = get_document_text(split_documents, parent_id, chunk_number)
                if len(text) > 20000:
                    text = text[:20000] + "..."
                document_entries.append(f"Document {i+1}:\n{text}\n")
            else:
                document_entries.append(f"Document {i+1}: [Missing metadata]\n")
        context_text = '\n'.join(document_entries)
        
        
        
        
        # batch_prompt = f"""
        # QUESTION: {question}
        
        # ### Relevance Assessment Criteria (Score 0–5):
        # Assign a score between 0 and 5 for each document based on how well it helps answer the question.

        # - 5 = Very relevant: directly answers the question or provides strong supporting information.
        # - 4 = Mostly relevant: discusses the topic and likely contributes to the answer.
        # - 3 = Somewhat relevant: may be useful but is incomplete or tangential.
        # - 2 = Weakly relevant: touches on the topic but doesn't help much.
        # - 1 = Barely relevant: loosely connected, mostly noise.
        # - 0 = Not relevant at all: completely unrelated to the question.

        # ### Instructions:
        # Please assign one score (0–5) per document.

        # CONTEXT:
        # {context_text}

        # ### Output Format:
        # Return a JSON array of scores:
        
        # I only need this, nothing else
        
        # ```json
        # [5, 4, 0, 3, ...]
        # ```
        # """

        
        
        batch_prompt = f"""
        QUESTION: {question}
        
        ### Relevance Assessment Criteria (Score 0–3):
        Assign a score between 0 and 3 for each document based on how well it helps answer the question.

        - 3 = Highly relevant: directly answers the question or provides essential information.
        - 2 = Moderately relevant: discusses the topic and contributes to the answer.
        - 1 = Slightly relevant: touches on the topic but doesnot contribute to the answer.
        - 0 = Not relevant: unrelated to the question or provides no useful information.

        ### Instructions:
        Please assign one score (0–3) per document.

        CONTEXT:
        {context_text}

        ### Output Format:
        Return a JSON array of scores:
        
        I only need this, nothing else
        
        ```json
        [3, 2, 0, 1, ...]
        ```
        """
    
        try:
            try:
                response = llm.invoke(batch_prompt)
                response_text = response.content if hasattr(response, 'content') else str(response)
                time.sleep(30)  
            except AttributeError:
                response = llm.chat.completions.create(
                    model=os.getenv("DEPLOYMENT_NAME", "gpt-4"),
                    messages=[
                        {"role": "system", "content": "You are a document relevance assessment assistant."},
                        {"role": "user", "content": batch_prompt}
                    ],
                    temperature=0.0
                )
                response_text = response.choices[0].message.content
                time.sleep(10)
            
            match = re.search(r'\[.*?\]', response_text, re.DOTALL)
            if match:
                try:
                    batch_scores = json.loads(match.group(0))
                    print(f"✅ Parsed scores: {batch_scores}")
                    if len(batch_scores) != len(batch_chunks):
                        print(f"⚠️ Score count mismatch: {len(batch_scores)} for {len(batch_chunks)} documents")
                        batch_scores += [5] * (len(batch_chunks) - len(batch_scores)) if len(batch_scores) < len(batch_chunks) else batch_scores[:len(batch_chunks)]
                except json.JSONDecodeError:
                    print("⚠️ Failed to parse JSON. Defaulting all scores to 5.")
                    batch_scores = [5] * len(batch_chunks)
            else:
                print("⚠️ No score list found. Using default scores.")
                batch_scores = [5] * len(batch_chunks)

            for i, (chunk, score) in enumerate(zip(batch_chunks, batch_scores)):
                try:
                    score = int(score)
                except:
                    score = 5
                doc_num = batch_idx + i + 1
                if score >3:
                    filtered_chunks.append(chunk)
                    print(f"Doc {doc_num} | Score: {score} ✅ KEPT")
                else:
                    print(f"Doc {doc_num} | Score: {score} ❌ DROPPED")
        except Exception as e:
            print(f"❌ Error during batch filtering: {e}")
            filtered_chunks.extend(batch_chunks)
            for i, chunk in enumerate(batch_chunks):
                doc_num = batch_idx + i + 1
                print(f"Doc {doc_num} | Score: 5 ⚠️ KEPT (Error)")

    duration = (datetime.now() - filter_start).total_seconds()
    print(f"⏱️ Filtering took {duration:.2f}s and kept {len(filtered_chunks)}/{len(chunks)} documents")
    
    return filtered_chunks

def process_range(start_idx: int, end_idx: int, batch_size: int, llm_batch_size: int, 
                 retrieval_file: str, split_file: str, filtered_output_file: str):
    # Load existing data FIRST
    existing_filtered = load_json_file(filtered_output_file, [])
    
    # Load source data
    retrieval_results = load_json_file(retrieval_file)
    if not retrieval_results:
        print(f"Error: No data found in {retrieval_file}")
        return
    
    # Load split documents
    split_documents = load_json_file(split_file)
    if not split_documents:
        print(f"Error: No data found in {split_file}")
        return
    
    # Create quick lookup maps for existing data
    filtered_map = {item['question_id']: item for item in existing_filtered}
    
    # Get LLM client
    llm = get_azure_llm()
    
    # Process each question from start_idx to end_idx
    for current_idx in range(start_idx, end_idx + 1):
        # Generate question ID to find
        question_id_to_find = f"Q{current_idx}"
        
        print(f"\nLooking for question {question_id_to_find}")
        
        # Skip if already processed
        if question_id_to_find in filtered_map:
            print(f"Skipping already processed question {question_id_to_find}")
            continue
        
        # Find the question with this ID
        target_query = None
        for query in retrieval_results:
            if query.get("question_id") == question_id_to_find:
                target_query = query
                break
        
        # Skip if not found
        if not target_query:
            print(f"Question {question_id_to_find} not found in retrieval results. Skipping.")
            continue
        
        print(f"Processing question {question_id_to_find}...")
        normal_chunks = target_query.get("Normal_retrieved_chunks", [])
        
        filtered_chunks = filter_documents_by_relevance(
            llm, target_query["question"], normal_chunks, split_documents, llm_batch_size
        )
        
        # Create new entry
        new_entry = {
            **target_query,
            "filtered_chunks": filtered_chunks,
        }
        
        # Store result
        filtered_map[question_id_to_find] = new_entry
        
        # Save after each question
        save_json_file(list(filtered_map.values()), filtered_output_file)
        print(f"Saved results for question {question_id_to_find}")

    print(f"\nFinal results:")
    print(f"- {len(filtered_map)} questions in {filtered_output_file}")

def main():
    print(f"Processing questions from Q{START_INDEX} to Q{END_INDEX}")
    process_range(
        START_INDEX,
        END_INDEX,
        BATCH_SIZE,
        LLM_BATCH_SIZE,
        RETRIEVAL_RESULTS_FILE,
        SPLIT_DOCUMENTS_FILE,
        FILTERED_OUTPUT_FILE
    )

if __name__ == "__main__":
    main()






# import json
# import re
# import os
# from datetime import datetime
# from typing import List, Dict, Any, Tuple
# from dotenv import load_dotenv
# import time

# # Load environment variables
# load_dotenv()

# # Configuration variables - MODIFY THESE AS NEEDED
# START_INDEX = 301
# END_INDEX = 400

# BATCH_SIZE = 1
# LLM_BATCH_SIZE = 30

# RETRIEVAL_RESULTS_FILE = "ret_results_1_500_Hybrid_15.json"
# SPLIT_DOCUMENTS_FILE = "split_documents_with_references.json"
# FILTERED_OUTPUT_FILE = "ret_results_filtered_Hybrid_15.json"

# def get_azure_llm():
#     try:
#         from generator import get_azure_llm as original_get_azure_llm
#         return original_get_azure_llm()
#     except ImportError:
#         from openai import AzureOpenAI
#         client = AzureOpenAI(
#             api_key=os.getenv("AZURE_OPENAI_API_KEY"),
#             api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
#             azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
#         )
#         return client

# def load_json_file(file_path: str, default_value=None) -> Any:
#     if not os.path.exists(file_path):
#         return default_value
#     try:
#         with open(file_path, 'r', encoding='utf-8') as f:
#             return json.load(f)
#     except Exception as e:
#         print(f"Error loading {file_path}: {e}")
#         return default_value

# def save_json_file(data: Any, file_path: str) -> None:
#     try:
#         with open(file_path, 'w', encoding='utf-8') as f:
#             json.dump(data, f, indent=4)
#         print(f"Successfully saved data to {file_path}")
#     except Exception as e:
#         print(f"Error saving to {file_path}: {e}")
#         raise

# def get_document_text(split_documents: List[Dict], parent_id: str, chunk_number: int) -> str:
#     for doc in split_documents:
#         if (doc.get("metadata", {}).get("parent_id") == parent_id and 
#             doc.get("metadata", {}).get("chunk_number") == chunk_number):
#             text = doc.get("text", "")
#             if isinstance(text, list):
#                 return " ".join(text)
#             return text
#     return f"[Document with parent_id {parent_id} and chunk_number {chunk_number} not found]"

# def filter_documents_by_relevance(llm, question: str, chunks: List[Dict], split_documents: List[Dict], llm_batch_size: int) -> List[Dict]:
#     if not chunks:
#         return []
    
#     filter_start = datetime.now()
#     filtered_chunks = []
    
#     for batch_idx in range(0, len(chunks), llm_batch_size):
#         batch_end = min(batch_idx + llm_batch_size, len(chunks))
#         batch_chunks = chunks[batch_idx:batch_end]
        
#         document_entries = []
#         for i, chunk in enumerate(batch_chunks):
#             parent_id = chunk.get("metadata", {}).get("parent_id")
#             chunk_number = chunk.get("metadata", {}).get("chunk_number")
#             if parent_id and chunk_number:
#                 text = get_document_text(split_documents, parent_id, chunk_number)
#                 if len(text) > 20000:
#                     text = text[:20000] + "..."
#                 document_entries.append(f"Document {i+1}:\n{text}\n")
#             else:
#                 document_entries.append(f"Document {i+1}: [Missing metadata]\n")
#         context_text = '\n'.join(document_entries)
        
#         batch_prompt = f"""
#         QUESTION: {question}
        
#         ### Relevance Assessment Criteria (Score 0–3):
#         Assign a score between 0 and 3 for each document based on how well it helps answer the question.

#         - 3 = Highly relevant: directly answers the question or provides essential information.
#         - 2 = Moderately relevant: discusses the topic and contributes to the answer.
#         - 1 = Slightly relevant: touches on the topic but doesnot contribute to the answer.
#         - 0 = Not relevant: unrelated to the question or provides no useful information.

#         ### Instructions:
#         Please assign one score (0–3) per document.

#         CONTEXT:
#         {context_text}

#         ### Output Format:
#         Return a JSON array of scores:
        
#         I only need this, nothing else
        
#         ```json
#         [3, 2, 0, 1, ...]
#         ```
#         """
        
#         try:
#             try:
#                 response = llm.invoke(batch_prompt)
#                 response_text = response.content if hasattr(response, 'content') else str(response)
#                 time.sleep(30)  
#             except AttributeError:
#                 response = llm.chat.completions.create(
#                     model=os.getenv("DEPLOYMENT_NAME", "gpt-4"),
#                     messages=[
#                         {"role": "system", "content": "You are a document relevance assessment assistant."},
#                         {"role": "user", "content": batch_prompt}
#                     ],
#                     temperature=0.0
#                 )
#                 response_text = response.choices[0].message.content
#                 time.sleep(10)
            
#             match = re.search(r'\[.*?\]', response_text, re.DOTALL)
#             if match:
#                 try:
#                     batch_scores = json.loads(match.group(0))
#                     print(f"✅ Parsed scores: {batch_scores}")
#                     if len(batch_scores) != len(batch_chunks):
#                         print(f"⚠️ Score count mismatch: {len(batch_scores)} for {len(batch_chunks)} documents")
#                         batch_scores += [3] * (len(batch_chunks) - len(batch_scores)) if len(batch_scores) < len(batch_chunks) else batch_scores[:len(batch_chunks)]
#                 except json.JSONDecodeError:
#                     print("⚠️ Failed to parse JSON. Defaulting all scores to 3.")
#                     batch_scores = [3] * len(batch_chunks)
#             else:
#                 print("⚠️ No score list found. Using default scores.")
#                 batch_scores = [3] * len(batch_chunks)

#             for i, (chunk, score) in enumerate(zip(batch_chunks, batch_scores)):
#                 try:
#                     score = int(score)
#                 except:
#                     score = 3
#                 doc_num = batch_idx + i + 1
#                 if score >= 2:
#                     filtered_chunks.append(chunk)
#                     print(f"Doc {doc_num} | Score: {score} ✅ KEPT")
#                 else:
#                     print(f"Doc {doc_num} | Score: {score} ❌ DROPPED")
#         except Exception as e:
#             print(f"❌ Error during batch filtering: {e}")
#             filtered_chunks.extend(batch_chunks)
#             for i, chunk in enumerate(batch_chunks):
#                 doc_num = batch_idx + i + 1
#                 print(f"Doc {doc_num} | Score: 3 ⚠️ KEPT (Error)")

#     duration = (datetime.now() - filter_start).total_seconds()
#     print(f"⏱️ Filtering took {duration:.2f}s and kept {len(filtered_chunks)}/{len(chunks)} documents")
    
#     return filtered_chunks

# def process_range(start_idx: int, end_idx: int, batch_size: int, llm_batch_size: int, 
#                  retrieval_file: str, split_file: str, filtered_output_file: str):
#     # Load existing data FIRST
#     existing_filtered = load_json_file(filtered_output_file, [])
    
#     # Load source data
#     retrieval_results = load_json_file(retrieval_file)
#     if not retrieval_results:
#         print(f"Error: No data found in {retrieval_file}")
#         return
    
#     # Load split documents
#     split_documents = load_json_file(split_file)
#     if not split_documents:
#         print(f"Error: No data found in {split_file}")
#         return
    
#     # Create quick lookup maps for existing data
#     filtered_map = {item['question_id']: item for item in existing_filtered}
    
#     # Get LLM client
#     llm = get_azure_llm()
    
#     # Process each question from start_idx to end_idx
#     for current_idx in range(start_idx, end_idx + 1):
#         # Generate question ID to find
#         question_id_to_find = f"Q{current_idx}"
        
#         print(f"\nLooking for question {question_id_to_find}")
        
#         # Skip if already processed
#         if question_id_to_find in filtered_map:
#             print(f"Skipping already processed question {question_id_to_find}")
#             continue
        
#         # Find the question with this ID
#         target_query = None
#         for query in retrieval_results:
#             if query.get("question_id") == question_id_to_find:
#                 target_query = query
#                 break
        
#         # Skip if not found
#         if not target_query:
#             print(f"Question {question_id_to_find} not found in retrieval results. Skipping.")
#             continue
        
#         print(f"Processing question {question_id_to_find}...")
#         normal_chunks = target_query.get("Normal_retrieved_chunks", [])
        
#         filtered_chunks = filter_documents_by_relevance(
#             llm, target_query["question"], normal_chunks, split_documents, llm_batch_size
#         )
        
#         # Create new entry
#         new_entry = {
#             **target_query,
#             "filtered_chunks": filtered_chunks,
#         }
        
#         # Store result
#         filtered_map[question_id_to_find] = new_entry
        
#         # Save after each question
#         save_json_file(list(filtered_map.values()), filtered_output_file)
#         print(f"Saved results for question {question_id_to_find}")

#     print(f"\nFinal results:")
#     print(f"- {len(filtered_map)} questions in {filtered_output_file}")

# def main():
#     print(f"Processing questions from Q{START_INDEX} to Q{END_INDEX}")
#     process_range(
#         START_INDEX,
#         END_INDEX,
#         BATCH_SIZE,
#         LLM_BATCH_SIZE,
#         RETRIEVAL_RESULTS_FILE,
#         SPLIT_DOCUMENTS_FILE,
#         FILTERED_OUTPUT_FILE
#     )

# if __name__ == "__main__":
#     main()