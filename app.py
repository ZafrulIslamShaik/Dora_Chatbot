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


st.set_page_config(
    page_title="DORA Regulation Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Times New Roman font and link styling
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
2. If the answer cannot be determined from the chunks, respond: "I don't know. The answer is not in the provided documents."
3. DO NOT use any prior knowledge beyond what's necessary to interpret the chunks.
4. After your answer, you MUST include a "CHUNKS USED:" section formatted exactly as shown:

#### **Answer:**
[Your answer text here]

**CHUNKS USED:**  
[1,2,3]

Replace the numbers with the actual chunk numbers you used in your answer (e.g., if you used chunks 1, 3, and 5, write [1,3,5]).

YOUR RESPONSE:"""
    
    return prompt

def detect_poor_quality_answer(answer):
    """Detect if answer indicates poor quality using multiple methods"""
    answer_lower = answer.lower()
    
    # Method 1: Character count - very short answers are suspicious
    if len(answer.strip()) < 75:
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
            return True
    
    # Method 3: Fuzzy matching for variations
    dont_know_variants = [
        "i don't know", "i do not know", "not sure", "cannot determine"
    ]
    
    for variant in dont_know_variants:
        similarity = SequenceMatcher(None, variant, answer_lower[:50]).ratio()
        if similarity > 0.8:  # 80% similarity threshold
            return True
    
    return False

def apply_cosine_filtering(retriever, question, retrieved_docs, threshold=0.5, use_filtering=False):
    """Apply cosine similarity filtering, controlled by UI toggle"""
    if not use_filtering or not hasattr(retriever, 'get_query_embedding'):
        return retrieved_docs
        
    query_embedding = retriever.get_query_embedding(question)
    filtered_docs = []
    
    for doc in retrieved_docs:
        # Use score if available, otherwise keep the document
        if hasattr(doc, 'metadata') and 'score' in doc.metadata:
            if doc.metadata['score'] >= threshold:
                filtered_docs.append(doc)
        else:
            filtered_docs.append(doc)
    
    return filtered_docs

def extract_chunk_numbers_from_answer(answer):
    """Extract chunk numbers from the answer in [1,2,3] format"""
    pattern = r'\[(\d+(?:,\d+)*)\]'
    match = re.search(pattern, answer)
    if match:
        try:
            # Parse the array format [1,2,3]
            numbers_str = '[' + match.group(1) + ']'
            chunk_numbers = json.loads(numbers_str)
            return chunk_numbers
        except json.JSONDecodeError:
            return []
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
        
        # Check if this chunk should be highlighted and auto-expanded (using 1-based indexing)
        chunk_number = i + 1
        is_target = st.session_state.target_chunk_number == chunk_number
        
        # Use expander with auto-open if targeted
        with st.expander(f"Chunk {chunk_number}", expanded=is_target):
            # Add highlight class if this is the target chunk
            if is_target:
                st.markdown('<div class="highlighted-chunk">', unsafe_allow_html=True)
            
            st.markdown(f"**Text:** {text}")
            st.markdown(f"**Chunk Number:** {chunk_number}")
            st.markdown(f"**Parent ID:** {parent_id}")
            st.markdown(f"**References:** {references}")
            
            if is_target:
                st.markdown('</div>', unsafe_allow_html=True)

def run_single_retrieval(question, retriever_type, k, alpha, use_hyde, use_llm_filtering, use_cosine_filtering, cosine_threshold, llm):
    """Run a single retrieval attempt"""
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
    
    # Retrieve documents - handle use_hyde parameter correctly for each retriever type
    if retriever_type in ["Dense_Retriever", "HyDE_Retriever"]:
        # Only DenseRetriever supports use_hyde
        retrieved_docs = retriever.get_relevant_documents(question, use_hyde=use_hyde)
    else:
        # SparseRetriever and HybridRetriever don't support use_hyde
        retrieved_docs = retriever.get_relevant_documents(question)
    
    # Store original docs
    all_docs = retrieved_docs.copy()
    
    # Apply LLM-based filtering if enabled from UI
    if use_llm_filtering:
        retrieved_docs = filter_documents_by_llm(retriever, question, retrieved_docs)
    
    # Apply cosine filtering if enabled from UI
    if use_cosine_filtering:
        retrieved_docs = apply_cosine_filtering(retriever, question, retrieved_docs, cosine_threshold, use_filtering=True)
    
    # Store all docs for display
    st.session_state.retrieved_chunks = all_docs
    
    # Generate answer
    prompt = create_answer_prompt(question, retrieved_docs)
    response = llm.invoke(prompt)
    answer = response.content if hasattr(response, 'content') else str(response)
    
    return answer, all_docs

def run_retrieval_pipeline(question, retriever_type, k, alpha=0.7, use_hyde=False, use_llm_filtering=False, use_cosine_filtering=False, cosine_threshold=0.5):
    """Main RAG pipeline with HyDE fallback"""
    start_time = datetime.now()
    

    llm = get_azure_llm()
    
    # First attempt - normal retrieval
    answer, all_docs = run_single_retrieval(
        question, retriever_type, k, alpha, use_hyde, 
        use_llm_filtering, use_cosine_filtering, cosine_threshold, llm
    )
    
    # Check if answer quality is poor and HyDE fallback is needed
    if detect_poor_quality_answer(answer) and not use_hyde and retriever_type != "HyDE_Retriever":
        print("Poor quality answer detected. Triggering HyDE fallback...")
        
        # Second attempt with HyDE enhancement
        answer_hyde, all_docs_hyde = run_single_retrieval(
            question, retriever_type, k, alpha, True,  # use_hyde=True
            use_llm_filtering, use_cosine_filtering, cosine_threshold, llm
        )
        
        # Use HyDE result if it's better quality
        if not detect_poor_quality_answer(answer_hyde):
            answer = answer_hyde
            all_docs = all_docs_hyde
            st.session_state.retrieved_chunks = all_docs_hyde
            print("HyDE fallback successful - using enhanced answer")
        else:
            print("HyDE fallback did not improve quality - using original answer")
    
    # Store the final answer
    st.session_state.last_answer = answer
    
    total_time = (datetime.now() - start_time).total_seconds()
    
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
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

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
    if retriever_type == "Hybrid_Retriever":
        alpha = st.sidebar.slider("Alpha (Dense Weight)", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
    
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
    
    cosine_filtering = st.sidebar.radio(
        "Cosine Similarity Filtering",
        options=["Off", "On"],
        index=0  
    )
    use_cosine_filtering = cosine_filtering == "On"
    
    # Cosine threshold only if cosine filtering is enabled
    cosine_threshold = 0.5
    if use_cosine_filtering:
        cosine_threshold = st.sidebar.slider("Cosine Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    
    # Toggle for showing retrieved chunks
    show_chunks = st.sidebar.radio(
        "Show Retrieved Chunks",
        options=["Off", "On"],
        index=0  # Default to Off
    )
    display_retrieved_chunks = show_chunks == "On"
    
    # Display chat history
    for i, message in enumerate(st.session_state.chat_history):
        with st.chat_message("user"):
            st.write(message["question"])
        with st.chat_message("assistant"):
            # Display answer with clickable links
            if i == len(st.session_state.chat_history) - 1:  # Only for most recent
                display_answer_with_links(message["answer"], i)
            else:
                st.write(message["answer"])
    
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
        
        # Generate response with spinner
        with st.chat_message("assistant"):
            with st.spinner("Generating answer..."):
                result = run_retrieval_pipeline(
                    question, 
                    retriever_type, 
                    k, 
                    alpha, 
                    use_hyde, 
                    use_llm_filtering, 
                    use_cosine_filtering, 
                    cosine_threshold
                )
                
                # Display the answer with clickable source links
                display_answer_with_links(result["answer"], len(st.session_state.chat_history))
        
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