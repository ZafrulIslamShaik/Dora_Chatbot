# app.py
import streamlit as st
import os
import json
import numpy as np
import re
from datetime import datetime
from difflib import SequenceMatcher
from dense_retriever import DenseRetriever
from sparse_retriever import SparseRetriever
from hybrid_retriever import HybridRetriever
from relevance_filtering import filter_documents_by_llm
from generator import get_azure_llm
from qdrant_client import QdrantClient
from document_embeddings.embedding_generator import get_azure_embedding
from llama_index.core.schema import Document


st.set_page_config(
    page_title="DORA Regulation Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

#Times New Roman font 
st.markdown("""
<style>
    * {
        font-family: 'Times New Roman', Times, serif !important;
    }
    .parent-id-link {
        color: #4361ee;
        text-decoration: underline;
        cursor: pointer;
    }
    .highlighted-chunk {
        border: 2px solid #4361ee;
        padding: 10px;
        background-color: #f0f8ff;
        animation: highlight-fade 3s;
    }
    @keyframes highlight-fade {
        from { background-color: #c9ddff; }
        to { background-color: #f0f8ff; }
    }
    .chunk-expander {
        margin-bottom: 10px;
    }
    .source-button {
        background-color: #4361ee !important;
        color: white !important;
        border: 2px solid #4361ee !important;
        padding: 8px 16px !important;
        font-weight: bold !important;
        border-radius: 8px !important;
        margin: 4px !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'retrieved_chunks' not in st.session_state:
    st.session_state.retrieved_chunks = []
if 'target_chunk_number' not in st.session_state:
    st.session_state.target_chunk_number = None
if 'last_answer' not in st.session_state:
    st.session_state.last_answer = ""
if 'pending_enhancement' not in st.session_state:
    st.session_state.pending_enhancement = None
if 'show_enhanced' not in st.session_state:
    st.session_state.show_enhanced = {}

def create_answer_prompt(question, context_chunks):
    """Create prompt for answer generation with citation format"""
    formatted_chunks = ""
    for i, chunk in enumerate(context_chunks):
        text = chunk.text if hasattr(chunk, 'text') else chunk.page_content
        
        formatted_chunks += f"CHUNK {i+1}:\n{text}\n\n"
    
    prompt = f"""You are a Digital Operational Resilience Act (DORA) expert assistant.

USER QUESTION: {question}

CONTEXT CHUNKS:
{formatted_chunks}

INSTRUCTIONS:
1. Answer ONLY using information from the context chunks provided above.
2. If the answer cannot be determined from the chunks, respond: "I don't know."
3. DO NOT use any prior knowledge beyond what's necessary to interpret the chunks.
4. After your answer, you MUST include a "CHUNKS USED:" section formatted exactly as shown:

#### **Answer:**
[Your answer text here]

**CHUNKS USED:**  
[1,2,3]

Replace the numbers with the actual chunk numbers you used in your answer (e.g., if you used chunks 1, 3, and 5, write [1,3,5]).

YOUR RESPONSE:"""
    
    return prompt

def calculate_cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    # Convert to numpy arrays
    a = np.array(vec1)
    b = np.array(vec2)
    
    # Calculate cosine similarity
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    similarity = dot_product / (norm_a * norm_b)
    return float(similarity)

def detect_poor_quality_answer(answer):
    """Detect if answer indicates poor quality using multiple methods"""
    print("\n" + "="*60)
    print("RAW LLM RESPONSE:")
    print("="*60)
    print(answer)
    print("="*60)
    
    answer_lower = answer.lower()
    
    # Method 1: Character count - very short answers are suspicious
    if len(answer.strip()) < 75:
        print(f"POOR QUALITY DETECTED: True (reason: too short - {len(answer.strip())} characters)")
        return True
    
    # Method 2: Direct keyword detection
    uncertainty_phrases = [
        "i don't know", "i do not know", "i'm not sure", "i am not sure",
        "cannot determine", "not clear", "unclear", "insufficient information",
        "not in the provided documents", "cannot be determined", "i'm unsure",
        "it's not evident", "the documents don't specify", "not specified"
    ]
    
    for phrase in uncertainty_phrases:
        if phrase in answer_lower:
            print(f"POOR QUALITY DETECTED: True (reason: contains uncertainty phrase '{phrase}')")
            return True
    
    # Method 3: Fuzzy matching for variations
    dont_know_variants = [
        "i don't know", "i do not know", "not sure", "cannot determine"
    ]
    
    for variant in dont_know_variants:
        similarity = SequenceMatcher(None, variant, answer_lower[:50]).ratio()
        if similarity > 0.8:  # 80% similarity threshold
            print(f"POOR QUALITY DETECTED: True (reason: fuzzy match with '{variant}' - {similarity:.2f} similarity)")
            return True
    
    print("POOR QUALITY DETECTED: False (answer seems good quality)")
    return False

def enhance_with_cross_references(retrieved_docs, question):
    """
    Enhance retrieved documents with cross-reference legal documents.
    NEW IMPLEMENTATION: Reference-first approach with cosine similarity ranking
    
    Args:
        retrieved_docs: List of documents from main retrieval
        question: User question for semantic search
        
    Returns:
        Enhanced list with original docs + top 5 cross-reference docs
    """
    print(f"\n" + "="*60)
    print("CROSS-REFERENCE ENHANCEMENT")
    print("="*60)
    
    try:
        # Connect to Qdrant cross-references collection
        qdrant_client = QdrantClient(host="localhost", port=6333)
        
        # Extract all references from retrieved documents
        all_references = []
        print(f"Input: {len(retrieved_docs)} cited chunks")
        print("Extracting references from cited chunks:")
        
        for i, doc in enumerate(retrieved_docs):
            references = doc.metadata.get("references", [])
            if isinstance(references, list) and references:
                print(f"  chunk{i+1}: {len(references)} references found")
                for ref in references:
                    if isinstance(ref, dict):
                        doc_name = ref.get("document", "")
                        article = ref.get("article_number", [])
                        paragraph = ref.get("Paragraph", [])
                        print(f"    - {doc_name}, Article: {article}, Paragraph: {paragraph}")
                all_references.extend(references)
            else:
                print(f"  chunk{i+1}: no references")
        
        if not all_references:
            print("❌ No references found in cited chunks")
            print("="*60)
            return retrieved_docs
        
        print(f"Total references to search: {len(all_references)}")
        print("-"*60)
        
        # Generate query embedding for similarity calculation
        print("Generating query embedding...")
        query_embedding = get_azure_embedding(question)
        
        # Get all documents from cross-references collection for reference matching
        print("Loading cross-references collection...")
        all_cross_refs = qdrant_client.scroll(
            collection_name="cross_references",
            limit=10000,  
            with_vectors=True 
        )[0]
        
        print(f"Loaded {len(all_cross_refs)} cross-reference documents")
        print("-"*60)
        
        # Find matching legal documents and calculate similarities
        matched_docs = []
        print("Searching for referenced legal documents:")
        
        for ref_idx, ref in enumerate(all_references):
            if not isinstance(ref, dict):
                continue
                
            document = ref.get("document", "")
            article_numbers = ref.get("article_number", [])
            paragraphs = ref.get("Paragraph", [])
            
            if not document:
                continue
            
            print(f"\nReference {ref_idx+1}: {document}")
            found_match = False
            
            # Priority 1: Try exact match (document + article + paragraph)
            if article_numbers and paragraphs:
                for article in article_numbers:
                    for paragraph in paragraphs:
                        target_ref = f"Document: {document}, Article: {article}, Paragraph: {paragraph}"
                        print(f"  Trying exact match: {target_ref}")
                        
                        for point in all_cross_refs:
                            stored_ref = point.payload.get("reference", "")
                            if target_ref in stored_ref or stored_ref in target_ref:
                                # Calculate cosine similarity
                                doc_vector = point.vector
                                similarity = calculate_cosine_similarity(query_embedding, doc_vector)
                                
                                cross_ref_doc = Document(
                                    text=point.payload.get("text", ""),
                                    metadata={
                                        **point.payload,
                                        "document_type": "cross_reference",
                                        "match_type": "exact",
                                        "similarity_score": similarity,
                                        "reference_match": target_ref
                                    }
                                )
                                matched_docs.append(cross_ref_doc)
                                print(f"    ✅ Found exact match! Similarity: {similarity:.3f}")
                                found_match = True
                                break
                    if found_match:
                        break
            
            # Priority 2: Try article fallback (document + article)
            if not found_match and article_numbers:
                for article in article_numbers:
                    target_ref = f"Document: {document}, Article: {article}"
                    print(f"  Trying article fallback: {target_ref}")
                    
                    for point in all_cross_refs:
                        stored_ref = point.payload.get("reference", "")
                        if target_ref in stored_ref:
                            # Calculate cosine similarity
                            doc_vector = point.vector
                            similarity = calculate_cosine_similarity(query_embedding, doc_vector)
                            
                            cross_ref_doc = Document(
                                text=point.payload.get("text", ""),
                                metadata={
                                    **point.payload,
                                    "document_type": "cross_reference",
                                    "match_type": "article",
                                    "similarity_score": similarity,
                                    "reference_match": target_ref
                                }
                            )
                            matched_docs.append(cross_ref_doc)
                            print(f"    ✅ Found article match! Similarity: {similarity:.3f}")
                            found_match = True
                            break
                if found_match:
                    break
            
            # Priority 3: Try document fallback
            if not found_match:
                target_ref = f"Document: {document}"
                print(f"  Trying document fallback: {target_ref}")
                
                for point in all_cross_refs:
                    stored_ref = point.payload.get("reference", "")
                    if document in stored_ref:
                        # Calculate cosine similarity
                        doc_vector = point.vector
                        similarity = calculate_cosine_similarity(query_embedding, doc_vector)
                        
                        cross_ref_doc = Document(
                            text=point.payload.get("text", ""),
                            metadata={
                                **point.payload,
                                "document_type": "cross_reference",
                                "match_type": "document",
                                "similarity_score": similarity,
                                "reference_match": target_ref
                            }
                        )
                        matched_docs.append(cross_ref_doc)
                        print(f"Found document match! Similarity: {similarity:.3f}")
                        found_match = True
                        break
            
            if not found_match:
                print(f"  No match found for {document}")
        
        print(f"\nTotal matched documents: {len(matched_docs)}")
        print("-"*60)
        
        # Sort by similarity score (highest first)
        matched_docs.sort(key=lambda x: x.metadata["similarity_score"], reverse=True)
        
        # Take top 5 cross-reference documents
        top_cross_refs = matched_docs[:5]
        
        print("Top 5 cross-references selected:")
        for i, doc in enumerate(top_cross_refs):
            similarity = doc.metadata["similarity_score"]
            match_type = doc.metadata["match_type"]
            reference = doc.metadata.get("reference", "")
            print(f"  {i+1}. {match_type.upper()} match - Similarity: {similarity:.3f}")
            print(f"     {reference[:80]}...")
        
        print("-"*60)
        print(f"ENHANCEMENT RESULT:")
        print(f"  Original cited chunks: {len(retrieved_docs)}")
        print(f"  Cross-references added: {len(top_cross_refs)}")
        print(f"  Total context for enhanced answer: {len(retrieved_docs) + len(top_cross_refs)} chunks")
        print("="*60)
        
        # Combine original docs with cross-references
        enhanced_docs = retrieved_docs + top_cross_refs
        
        return enhanced_docs
        
    except Exception as e:
        print(f"Error in cross-reference enhancement: {e}")
        print("="*60)
        return retrieved_docs

def extract_chunk_numbers_from_answer(answer):
    pattern = r'\[(.*?)\]'
    match = re.search(pattern, answer)
    
    if match:
        inside = match.group(1)
        numbers = [int(x.strip()) for x in inside.split(',') if x.strip().isdigit()]
        return numbers
    
    return []


def display_chunks():
    """Display retrieved chunks with metadata, showing full text"""
    if not st.session_state.retrieved_chunks:
        st.info("No chunks retrieved yet.")
        return
    
    for i, chunk in enumerate(st.session_state.retrieved_chunks):
        text = chunk.text if hasattr(chunk, 'text') else chunk.page_content
        parent_id = chunk.metadata.get("parent_id", "unknown-id")
        references = chunk.metadata.get("references", "")
        
        # Check if this is a cross-reference chunk
        is_cross_ref = chunk.metadata.get("document_type") == "cross_reference"
        chunk_label = f"Cross-Reference {i+1}" if is_cross_ref else f"Chunk {i+1}"
        
        chunk_number = i + 1
        is_target = st.session_state.target_chunk_number == chunk_number
        with st.expander(chunk_label, expanded=is_target):
            if is_target:
                st.markdown('<div class="highlighted-chunk">', unsafe_allow_html=True)
            
            st.markdown(f"**Text:** {text}")
            st.markdown(f"**Chunk Number:** {chunk_number}")
            
            if is_cross_ref:
                st.markdown(f"**Type:** Cross-Reference Document")
                st.markdown(f"**Match Type:** {chunk.metadata.get('match_type', 'N/A')}")
                st.markdown(f"**Reference:** {chunk.metadata.get('reference', 'N/A')}")
            else:
                st.markdown(f"**Parent ID:** {parent_id}")
                st.markdown(f"**References:** {references}")
            
            if is_target:
                st.markdown('</div>', unsafe_allow_html=True)


def run_single_retrieval(question, retriever_type, k, alpha, use_hyde, use_llm_filtering, llm):
    """Run a single retrieval attempt - NO enhanced answer here"""
    print(f"\n" + "="*60)
    print("RETRIEVAL PROCESS")
    print("="*60)
    print(f"Retriever: {retriever_type}")
    print(f"K: {k}")
    print(f"Question: {question[:100]}...")
    print("-"*60)
    
    # Initialize retriever based on selection
    if retriever_type == "Dense_Retriever":
        retriever = DenseRetriever(k=k, llm=llm)
    elif retriever_type == "Sparse_Retriever":
        retriever = SparseRetriever(sparse_k=k)
    elif retriever_type == "Hybrid_Retriever":
        retriever = HybridRetriever(alpha=alpha, top_k=k)
    else:  # Dense with HyDE
        retriever = DenseRetriever(k=k, llm=llm)
        use_hyde = True
    
    if retriever_type in ["Dense_Retriever", "HyDE_Retriever"]:
        retrieved_docs = retriever.get_relevant_documents(question, use_hyde=use_hyde)
    else:
        retrieved_docs = retriever.get_relevant_documents(question)
    
    # Display initial retrieval results
    print(f"\nINITIAL RETRIEVAL RESULTS:")
    print(f"Retrieved {len(retrieved_docs)} chunks")
    

    all_docs = retrieved_docs.copy()
    
    # Apply LLM-based filtering if enabled from UI
    if use_llm_filtering:
        print(f"\nAPPLYING LLM FILTERING...")
        before_count = len(retrieved_docs)
        retrieved_docs = filter_documents_by_llm(retriever, question, retrieved_docs)
        after_count = len(retrieved_docs)
        
        # EXACTLY AS YOU REQUESTED: Log filtering results
        print(f"LLM filtering applied")
        print(f"Initial  : {before_count} chunks")
        print(f"After Filtering: {after_count} chunks")
        
        # Log kept chunks as you requested
        print("\nKept chunks:")
        for i, doc in enumerate(retrieved_docs):
            parent_id = doc.metadata.get('parent_id', 'unknown')
            score = doc.metadata.get('score', 'N/A')
            text_snippet = doc.text[:50] + '...' if doc.text else ''
            print(f"[Chunk {i+1}]")
            print(f"Parent ID: {parent_id}")
            print(f"Score: {score}")
            print(f"Text: {text_snippet}")
            print()
    
    # Store ONLY FILTERED DOCS in session state
    st.session_state.retrieved_chunks = retrieved_docs
    
    print(f"\nFINAL CONTEXT FOR LLM: {len(retrieved_docs)} chunks")
    print("="*60)
    
    prompt = create_answer_prompt(question, retrieved_docs)
    response = llm.invoke(prompt)
    answer = response.content if hasattr(response, 'content') else str(response)
    
    return answer, retrieved_docs, retriever


def generate_enhanced_answer(question, initial_answer, filtered_docs):
    """Generate enhanced answer using only cited chunks and their cross-references"""
    print("\n" + "🔥"*20)
    print("GENERATING ENHANCED ANSWER")
    print("🔥"*20)
    
    llm = get_azure_llm()
    
    # Extract cited chunk numbers from the initial answer
    cited_chunk_numbers = extract_chunk_numbers_from_answer(initial_answer)
    
    if not cited_chunk_numbers:
        print("No chunk citations found in initial answer")
        return None
    
    print(f"Cited chunks from initial answer: {cited_chunk_numbers}")

    cited_chunks = []
    for chunk_num in cited_chunk_numbers:
        if 1 <= chunk_num <= len(filtered_docs):
            cited_chunks.append(filtered_docs[chunk_num - 1])
            print(f"  Adding chunk {chunk_num} to cited chunks")
    
    print(f"Total cited chunks: {len(cited_chunks)}")
    
    if not cited_chunks:
        print("No valid cited chunks found")
        return None
    
    # Get cross-references based on ONLY the cited chunks
    enhanced_docs = enhance_with_cross_references(cited_chunks, question)
    
    # Update session state with the enhanced chunks
    st.session_state.retrieved_chunks = enhanced_docs
    
    # Generate new answer with enhanced prompt
    print("\n" + "💡"*20)
    print("GENERATING ENHANCED ANSWER")
    print("💡"*20)
    print(f"Using enhanced prompt with {len(enhanced_docs)} chunks")
    print(f"  - Cited chunks: {len(cited_chunks)}")
    print(f"  - Cross-references: {len(enhanced_docs) - len(cited_chunks)}")
    
    # Use the enhanced prompt for comprehensive answer
    enhanced_prompt = create_enhanced_answer_prompt(question, enhanced_docs)
    enhanced_response = llm.invoke(enhanced_prompt)
    answer = enhanced_response.content if hasattr(enhanced_response, 'content') else str(enhanced_response)
    
    print(f"Enhanced answer generated successfully!")
    print(f"Answer length: {len(answer)} characters")
    
    return answer

def create_enhanced_answer_prompt(question, context_chunks):
    """Create enhanced prompt for comprehensive answer with cross-references"""
    # Separate primary chunks from cross-references
    primary_chunks = []
    cross_ref_chunks = []
    
    for i, chunk in enumerate(context_chunks):
        if chunk.metadata.get("document_type") == "cross_reference":
            cross_ref_chunks.append((i+1, chunk))
        else:
            primary_chunks.append((i+1, chunk))
    
    # Format primary chunks
    formatted_primary = ""
    for idx, chunk in primary_chunks:
        text = chunk.text if hasattr(chunk, 'text') else chunk.page_content
        formatted_primary += f"PRIMARY CHUNK {idx}:\n{text}\n\n"
    
    # Format cross-reference chunks
    formatted_cross_refs = ""
    for idx, chunk in cross_ref_chunks:
        text = chunk.text if hasattr(chunk, 'text') else chunk.page_content
        reference = chunk.metadata.get("reference", "Unknown reference")
        formatted_cross_refs += f"CROSS-REFERENCE {idx} ({reference}):\n{text}\n\n"
    
    prompt = f"""You are a Regulatory Compliance Expert specialized in EU legislation,
particularly the Digital Operational Resilience Act (DORA)
and its related legal documents.

USER QUESTION: {question}

PRIMARY DORA CONTEXT:
{formatted_primary}

RELATED LEGAL CROSS-REFERENCES:
{formatted_cross_refs}

1. Answer the following QUESTION only using the provided CONTEXT.
2. You are not allowed to add any points that are not present in the context.
3. If there are any contradictions, reply with "I don’t know"


YOUR RESPONSE:"""
    
    return prompt

def run_retrieval_pipeline(question, retriever_type, k, alpha=0.7, use_hyde=False, use_llm_filtering=False):
    """Main RAG pipeline - only generates initial answer"""
    start_time = datetime.now()
    
    print(f"\n" + "🚀"*30)
    print("STARTING RAG PIPELINE")
    print("🚀"*30)
    print(f"Settings:")
    print(f"  - Retriever: {retriever_type}")
    print(f"  - K: {k}")
    print(f"  - LLM Filtering: {use_llm_filtering}")
    print(f"  - HyDE: {use_hyde}")

    llm = get_azure_llm()
    
    # STEP 1: First attempt - normal retrieval
    print("\n" + "📝"*20)
    print("INITIAL RETRIEVAL AND ANSWER GENERATION")
    print("📝"*20)
    
    answer, filtered_docs, retriever = run_single_retrieval(
        question, retriever_type, k, alpha, use_hyde, 
        use_llm_filtering, llm
    )
    
    # Check if answer quality is poor and HyDE fallback is needed
    if detect_poor_quality_answer(answer) and not use_hyde and retriever_type != "HyDE_Retriever":
        print("\n🔄 Poor quality answer detected. Triggering HyDE fallback...")
        
        # Second attempt with HyDE enhancement
        answer_hyde, filtered_docs_hyde, retriever_hyde = run_single_retrieval(
            question, retriever_type, k, alpha, True,  # use_hyde=True
            use_llm_filtering, llm
        )
        
        # Use HyDE result if it's better quality
        if not detect_poor_quality_answer(answer_hyde):
            answer = answer_hyde
            filtered_docs = filtered_docs_hyde
            retriever = retriever_hyde
            st.session_state.retrieved_chunks = filtered_docs_hyde
            print("HyDE fallback successful - using enhanced answer")
        else:
            print("HyDE fallback did not improve quality")
    
    # Store the final answer and filtered docs for potential enhancement
    st.session_state.last_answer = answer
    st.session_state.pending_enhancement = {
        'question': question,
        'initial_answer': answer,
        'filtered_docs': filtered_docs
    }
    
    total_time = (datetime.now() - start_time).total_seconds()
    

    print("PIPELINE COMPLETED")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Final answer length: {len(answer)} characters")
    print(f"Chunks in session state: {len(st.session_state.retrieved_chunks)}")

    
    return {
        "answer": answer,
        "processing_time": total_time
    }

def display_answer_with_links(answer, message_index):
    """Display answer with clickable source buttons instead of chunk numbers"""
    # Extract all chunk numbers mentioned in the answer
    chunk_numbers = extract_chunk_numbers_from_answer(answer)
    
    # Split answer into parts before and after the "CHUNKS USED:" section
    parts = answer.split("**CHUNKS USED:**")
    if len(parts) != 2:
        # If not found, just display the answer as is
        st.write(answer)
        return
    
    # Display the main answer part
    st.write(parts[0])
    
    # Display the "CHUNKS USED:" header with enhanced styling
    st.markdown("**📚 SOURCES USED:**", unsafe_allow_html=True)
    
    # Create numbered source buttons with enhanced styling
    for i, chunk_num in enumerate(chunk_numbers):
        # Create a unique key for each button
        button_key = f"link_chunk_{chunk_num}_{i}_{message_index}"
        
        # Show "Source N" button with enhanced styling
        if st.button(f"📄 Source {chunk_num}", key=button_key, 
                    help=f"Click to view source chunk {chunk_num}",
                    type="primary"):
            st.session_state.target_chunk_number = chunk_num
            st.rerun()  # Force a rerun to update the UI

def main():
    # Main title
    st.title("DORA Regulation Assistant")
    
    # Sidebar for configuration
    st.sidebar.title("Configuration")
    
    # Retriever selection
    retriever_type = st.sidebar.radio(
        "Select Retriever",
        ["Dense_Retriever", "Sparse_Retriever", "Hybrid_Retriever", "HyDE_Retriever"]
    )
    
    # Dynamic K parameter label based on retriever type
    k_label = f"{retriever_type.split('_')[0]} K"
    k = st.sidebar.slider(k_label, min_value=1, max_value=50, value=10)
    
    # Alpha for Hybrid retriever
    alpha = 0.7
    
    # HyDE is automatically used with HyDE_Retriever
    use_hyde = retriever_type == "HyDE_Retriever"
    
    # Toggle buttons for filtering options (default OFF)
    st.sidebar.subheader("Enhancement Options")
    
    # Use radio buttons as toggle switches with default OFF
    llm_filtering = st.sidebar.radio(
        "LLM Relevancy Filtering",
        options=["Off", "On"],
        index=0  
    )
    use_llm_filtering = llm_filtering == "On"
    
    # Toggle for showing retrieved chunks
    show_chunks = st.sidebar.radio(
        "Show Retrieved Chunks",
        options=["Off", "On"],
        index=0 
    )
    display_retrieved_chunks = show_chunks == "On"
    
    # Display chat history
    for i, message in enumerate(st.session_state.chat_history):
        with st.chat_message("user"):
            st.write(message["question"])
        with st.chat_message("assistant"):
            # Check if we should show enhanced answer
            if st.session_state.show_enhanced.get(i, False) and 'enhanced_answer' in message:
                # Show enhanced answer
                st.info("🔥 **Enhanced Answer with Cross-References:**")
                display_answer_with_links(message["enhanced_answer"], f"enhanced_{i}")
            else:
                # Show initial answer
                display_answer_with_links(message["answer"], i)
                
                # Show enhance button if this is the most recent message and no enhanced answer yet
                if i == len(st.session_state.chat_history) - 1 and 'enhanced_answer' not in message:
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        if st.button("🔥 Get Enhanced Answer with Cross-References", 
                                   key=f"enhance_btn_{i}",
                                   help="Generate a more comprehensive answer using cross-referenced legal documents"):
                            
                            # Generate enhanced answer
                            if st.session_state.pending_enhancement:
                                with st.spinner("Generating enhanced answer..."):
                                    enhanced_answer = generate_enhanced_answer(
                                        st.session_state.pending_enhancement['question'],
                                        st.session_state.pending_enhancement['initial_answer'],
                                        st.session_state.pending_enhancement['filtered_docs']
                                    )
                                    
                                    if enhanced_answer:
                                        # Update the message with enhanced answer
                                        st.session_state.chat_history[i]['enhanced_answer'] = enhanced_answer
                                        st.session_state.show_enhanced[i] = True
                                        st.rerun()
                                    else:
                                        st.error("Could not generate enhanced answer. No citations found in the initial answer.")
                
                # Show toggle button if enhanced answer exists
                elif 'enhanced_answer' in message:
                    if st.button("📋 Show Initial Answer", key=f"show_initial_{i}"):
                        st.session_state.show_enhanced[i] = False
                        st.rerun()
    
    # Show retrieved chunks if enabled
    if display_retrieved_chunks:
        st.subheader("Retrieved Chunks")
        display_chunks()
        # Reset target ID after displaying chunks
        if st.session_state.target_chunk_number:
            st.session_state.target_chunk_number = None
    
    # User input
    question = st.chat_input("Ask about DORA regulations...")
    
    if question:
        # Add user message to chat
        with st.chat_message("user"):
            st.write(question)
        
        # Generate response 
        with st.chat_message("assistant"):
            with st.spinner("Generating answer..."):
                result = run_retrieval_pipeline(
                    question, 
                    retriever_type, 
                    k, 
                    alpha, 
                    use_hyde, 
                    use_llm_filtering
                )
                
                # Display the answer with clickable source links
                display_answer_with_links(result["answer"], len(st.session_state.chat_history))
                
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button("🔥 Get Enhanced Answer with Cross-References", 
                               key="enhance_btn_current",
                               help="Generate a more comprehensive answer using cross-referenced legal documents"):
                        pass  # This will be handled on rerun
        
        # Save to history
        st.session_state.chat_history.append({
            "question": question,
            "answer": result["answer"]
        })
        
        # Show retrieved chunks after generation if enabled
        if display_retrieved_chunks:
            st.subheader("Retrieved Chunks")
            display_chunks()

if __name__ == "__main__":
    main()