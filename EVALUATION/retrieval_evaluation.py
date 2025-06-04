

"""
Retrieval Evaluation Pipeline

Comprehensive evaluation system for document retrieval methods including sparse, dense,HyDE
and hybrid approaches with optional reranking, and LLM filtering capabilities.

Processes question-context pairs in batches and measures retrieval performance.

"""


import json
import logging
import os
from Models.DORA_CHATBOT.EVALUATION.config import *
from datetime import datetime
from generator import get_azure_llm
from hyde import get_dora_hyde_answer
from dotenv import load_dotenv
from relevance_filtering import filter_documents_by_llm
from sparse_retriever import SparseRetriever
from dense_retriever import DenseRetriever
import os
import glob
cwd = os.getcwd()

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("azure").setLevel(logging.WARNING)

def load_chunks_from_file(chunks_file):
    """Load chunks from file.""" 
    try:
        with open(chunks_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        return chunks
    except Exception as e:
        raise
    
def get_retrieval_results_filename(batch_num, k_value, prompt_type='default'):
    timestamp = datetime.now().strftime("%H-%M-%S_%d-%m")
    results_dir = os.path.join(cwd, OUTPUT_FOLDER)
    os.makedirs(results_dir, exist_ok=True)
    return os.path.join(results_dir, f"ret_results.json")

def initialize_retriever_with_chunks(chunks, k, use_gpu=True, llm=None):
    VECTOR_EMBEDDINGS_FOLDER = "document_embeddings"
    embeddings_path = os.path.join(VECTOR_EMBEDDINGS_FOLDER, "all_embeddings.npy")

    from hybrid_retriever import HybridRetriever
    
    RetrieverClass = {
        "Sparse_Retriever": SparseRetriever,   
        "Dense_Retriever": DenseRetriever,
        "Hybrid_Retriever": HybridRetriever 
    }.get(RETRIEVER_TYPE, SparseRetriever)

    if RETRIEVER_TYPE == "Sparse_Retriever":
        return SparseRetriever(texts_path=CHUNKS_FILE, sparse_k=k)

    if RETRIEVER_TYPE == "Hybrid_Retriever":
        return HybridRetriever(
            alpha=ALPHA,
            top_k=k 
        )

    return RetrieverClass(
        embeddings_path=embeddings_path,
        texts_path=CHUNKS_FILE,
        k=k,
        llm=llm
    )
    
def evaluate_retrieval(retriever, question_context_pairs, k, use_hyde=False, hyde_answers=None, use_reranking=False, rerank_top_k=None):
    """
    Evaluate retrieval performance on question-context pairs.
    
    Args:
        retriever: The retriever to evaluate
        question_context_pairs: List of question-context pairs
        k: Number of documents to retrieve
        use_hyde: Whether to use HyDE
        hyde_answers: Pre-generated HyDE answers dictionary (question -> answer)
        use_reranking: Whether to use reranking
        rerank_top_k: Number of documents to rerank
    """
    retrieval_results = []
    
    
    for idx, pair in enumerate(question_context_pairs, 1):
        question = pair["question"]
        ground_truth = pair["context"]
        ground_truth_chunk_number = pair["chunk_number"]
        ground_truth_parent_id = pair.get("parent_id", "unknown")  
        
        hyde_answer = None
        if use_hyde:
            if hyde_answers and question in hyde_answers:
                hyde_answer = hyde_answers[question]
            else:
                llm = get_azure_llm()
                if hasattr(retriever, 'llm') and retriever.llm is None:
                    retriever.llm = llm
                if hasattr(retriever, 'hyde_retriever') and retriever.hyde_retriever.llm is None:
                    retriever.hyde_retriever.llm = llm
                    
                hyde_answer = get_dora_hyde_answer(llm, question)
        

        if RETRIEVER_TYPE == "Hybrid_Retriever":
            retrieved_docs = retriever.get_relevant_documents(question)
        elif RETRIEVER_TYPE == "Dense_Retriever":
            if use_hyde and hyde_answer:
                retrieved_docs = retriever.get_relevant_documents(hyde_answer, use_hyde=True)
            else:
                retrieved_docs = retriever.get_relevant_documents(question, use_hyde=use_hyde)
        else:
            retrieved_docs = retriever.get_relevant_documents(question)


        normal_parent_ids = []
        for doc in retrieved_docs:
            parent_id = None
            if hasattr(doc, 'metadata'):
                parent_id = doc.metadata.get('parent_id', "unknown")
            
            if parent_id is not None:
                normal_parent_ids.append(parent_id)

        normal_docs = retrieved_docs
        
        if not use_reranking and rerank_top_k is not None:
            normal_docs = normal_docs[:rerank_top_k]
            normal_parent_ids = normal_parent_ids[:rerank_top_k]
        
        reranked_docs = []
        filtered_docs = []
        
        normal_status = "HIT" if ground_truth_parent_id in normal_parent_ids else "MISS"
        rerank_status = "N/A"
        filter_status = "N/A"

        if use_reranking:
            try:
                from llama_index.core.postprocessor import SentenceTransformerRerank
                from llama_index.core.schema import NodeWithScore, TextNode, Document  
                

                llama_nodes = []
                for doc in normal_docs:

                    text_node = TextNode(
                        text=doc.text,
                        metadata=doc.metadata
                    )

                    node_with_score = NodeWithScore(
                        node=text_node,
                        score=1.0  
                    )
                    llama_nodes.append(node_with_score)
                

                reranker = SentenceTransformerRerank(
                    model="cross-encoder/ms-marco-MiniLM-L-6-v2",
                    top_n=rerank_top_k
                )
                

                reranked_nodes = reranker.postprocess_nodes(
                    nodes=llama_nodes,
                    query_str=question
                )
                

                reranked_docs = []
                for node_with_score in reranked_nodes:
                    text_node = node_with_score.node
                    doc = Document(
                        text=text_node.text,
                        metadata=text_node.metadata
                    )
                    reranked_docs.append(doc)
                

                reranked_parent_ids = []
                for doc in reranked_docs:
                    parent_id = doc.metadata.get('parent_id', "unknown")
                    reranked_parent_ids.append(parent_id)
                
                rerank_status = "HIT" if ground_truth_parent_id in reranked_parent_ids else "MISS"
            except Exception as e:
                print(f"Error using LlamaIndex reranker: {e}")
                reranked_docs = []
                reranked_parent_ids = []
                rerank_status = "ERROR"
                

        filtered_docs = []
        filtered_parent_ids = []
        
        if USE_LLM_FILTERING:
            filtered_docs = filter_documents_by_llm(retriever, question, normal_docs)
            

            filtered_parent_ids = []
            for doc in filtered_docs:
                parent_id = None
                if hasattr(doc, 'metadata'):
                    parent_id = doc.metadata.get('parent_id', "unknown")
                
                if parent_id is not None:
                    filtered_parent_ids.append(parent_id)
            
            filter_status = "HIT" if ground_truth_parent_id in filtered_parent_ids else "MISS"


        if RETRIEVER_TYPE == "hybrid" and hasattr(retriever, 'sparse_retriever') and hasattr(retriever, 'dense_retriever'):
            sparse_docs = retriever.sparse_retriever.get_relevant_documents(question)
            if USE_HYDE and retriever.llm:
                if hyde_answers and question in hyde_answers:
                    dense_docs = retriever.dense_retriever.get_relevant_documents(hyde_answers[question], use_hyde=True)
                else:
                    hyde_answer = get_dora_hyde_answer(retriever.llm, question)
                    dense_docs = retriever.dense_retriever.get_relevant_documents(hyde_answer, use_hyde=True)
            else:
                dense_docs = retriever.dense_retriever.get_relevant_documents(question)
            
            combined_docs_with_metadata = []
            seen_contents = set()
            for doc in sparse_docs + dense_docs:
                content = doc.text  
                if content not in seen_contents:
                    seen_contents.add(content)
                    metadata = {}
                    if hasattr(doc, "metadata"):
                        metadata = {
                            "parent_id": doc.metadata.get("parent_id", None),
                            "chunk_number": doc.metadata.get("chunk_number", None)
                        }
                    combined_docs_with_metadata.append({
                        "metadata": metadata
                    })
        else:

            combined_docs_with_metadata = []
            for doc in normal_docs:
                metadata = {}
                if hasattr(doc, "metadata"):
                    metadata = {
                        "parent_id": doc.metadata.get("parent_id", None),
                        "chunk_number": doc.metadata.get("chunk_number", None)
                    }
                combined_docs_with_metadata.append({
                    "metadata": metadata
                })

        # Create retrieval results
        result_entry = {
            "question": question,
            "parent_id": ground_truth_parent_id,
            "question_id": pair["question_id"],
            "Normal": normal_status,
            "Normal_retrieved_chunks": combined_docs_with_metadata
        }
            
        if use_reranking:
            reranked_docs_with_metadata = []
            for doc in reranked_docs:
                metadata = {}
                if hasattr(doc, "metadata"):
                    metadata = {
                        "parent_id": doc.metadata.get("parent_id", None),
                        "chunk_number": doc.metadata.get("chunk_number", None)
                    }
                reranked_docs_with_metadata.append({
                    "metadata": metadata
                })
            result_entry["reranked_chunks"] = reranked_docs_with_metadata
            result_entry["Postrerank"] = rerank_status
            
        if USE_LLM_FILTERING:
            filtered_docs_with_metadata = []
            for doc in filtered_docs:
                metadata = {}
                if hasattr(doc, "metadata"):
                    metadata = {
                        "parent_id": doc.metadata.get("parent_id", None),
                        "chunk_number": doc.metadata.get("chunk_number", None)
                    }
                filtered_docs_with_metadata.append({
                    "metadata": metadata
                })
            result_entry["filtered_chunks"] = filtered_docs_with_metadata
            filter_stage = "Postfilter"
            result_entry[filter_stage] = filter_status
          
        retrieval_results.append(result_entry)

    results = {
        "retrieval_results": retrieval_results
    }
  
    return results

def get_next_batch_number():
    """Find the next batch number by checking existing files"""
    results_dir = os.path.join(cwd, OUTPUT_FOLDER)
    os.makedirs(results_dir, exist_ok=True)
    
    batch_files = glob.glob(os.path.join(results_dir, "ret_results_batch_*.json"))
    
    if not batch_files:
        return 1
    
    batch_numbers = []
    for file in batch_files:
        filename = os.path.basename(file)
        try:
            num = int(filename.replace("ret_results_batch_", "").split("_")[0].split(".")[0])
            batch_numbers.append(num)
        except ValueError:
            continue
    
    if not batch_numbers:
        return 1
    
    # Return the next batch number
    return max(batch_numbers) + 1

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )
    questions_file = output_file
    
    print(questions_file)
    
    with open(questions_file, 'r') as f:
        all_questions = json.load(f)
        total_questions = len(all_questions[START_INDEX-1:END_INDEX])
        print(f"Total questions to process: {total_questions}")
    
    llm = None
    if USE_HYDE:
        from generator import get_azure_llm
        llm = get_azure_llm()
    
    chunks = load_chunks_from_file(CHUNKS_FILE)
    
    # Process each batch
    for start_idx in range(START_INDEX, END_INDEX + 1, BATCH_SIZE):
        end_idx = min(start_idx + BATCH_SIZE - 1, END_INDEX)
        
        batch_number = get_next_batch_number()
        
        print(f"Processing batch {batch_number}: questions {start_idx} to {end_idx}")
        
        question_context_pairs = all_questions[start_idx-1:end_idx]
        
        hyde_answers = {}
        if USE_HYDE:
            print(f"Pre-generating hypothetical documents for batch {batch_number}...")
            
            for pair in question_context_pairs:
                question = pair["question"]
                if question not in hyde_answers:
                    hyde_answers[question] = get_dora_hyde_answer(llm, question)
                    print(f"Generated HyDE answer for: {question[:50]}...")
            
            print(f"Generated {len(hyde_answers)} hypothetical documents")
        
        for k in K_VALUES:
            print(f"Processing with k={k}...")

            retriever = initialize_retriever_with_chunks(chunks, k, llm=llm)

            results = evaluate_retrieval(
                retriever, 
                question_context_pairs, 
                k, 
                use_hyde=USE_HYDE,
                hyde_answers=hyde_answers if USE_HYDE else None,
                use_reranking=USE_RERANKING, 
                rerank_top_k=RERANK_TOP_K
            )
    
            retrieval_results_filename = get_retrieval_results_filename(batch_number, k)
            print(f"Writing retrieval results to {retrieval_results_filename}")
            
            # Save retrieval results
            with open(retrieval_results_filename, "w") as f:
                json.dump(results["retrieval_results"], f, indent=4)
            
        print(f"Completed batch {batch_number}: questions {start_idx} to {end_idx}")