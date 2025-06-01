import os
import sys
import numpy as np
import json
import time
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, OptimizersConfigDiff
from qdrant_client.http.exceptions import UnexpectedResponse

# === CONFIGURATION ===
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333

COLLECTION_NAME = "dora_embeddings"
EMBEDDING_FILE = "document_embeddings/split_documents.npy"
CHUNKS_FILE = "document_embeddings/split_documents.json"  

BATCH_SIZE = 100

def log(message):
    """Helper function for logging with timestamp"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def main():
    log("Starting migration from FAISS to Qdrant")

    # Connect to Qdrant
    log(f"Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    
    # Check if collection exists and delete if needed
    try:
        collection_info = client.get_collection(COLLECTION_NAME)
        log(f"Collection '{COLLECTION_NAME}' already exists. Recreating...")
        client.delete_collection(COLLECTION_NAME)
    except UnexpectedResponse:
        log(f"Collection '{COLLECTION_NAME}' does not exist. Creating new.")

    # Load embeddings
    log(f"Loading embeddings from {EMBEDDING_FILE}")
    embeddings = np.load(EMBEDDING_FILE)
    vector_size = embeddings.shape[1]
    total_vectors = embeddings.shape[0]
    
    with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    payload_list = []
    for chunk in chunks:
        text_content = ""
        if "text" in chunk:
            if isinstance(chunk["text"], list) and chunk["text"]:
                text_content = chunk["text"][0] 
            else:
                text_content = chunk.get("text", "")
                
        # Create payload with text and metadata
        payload = {
            "text": text_content,
            **chunk.get("metadata", {})  # Include all metadata fields
        }
        payload_list.append(payload)
    
    # Validate chunk count
    if len(payload_list) != total_vectors:
        log(f"ERROR: Mismatch between embeddings ({total_vectors}) and chunks ({len(payload_list)})!")
        sys.exit(1)
    
    # Create collection with L2 distance (Euclidean) to match FAISS IndexFlatL2
    log(f"Creating collection '{COLLECTION_NAME}' with vector size {vector_size}")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.EUCLID  
        ),
        optimizers_config=OptimizersConfigDiff(
            indexing_threshold=20000  
        )
    )
    
    # Upload vectors in batches
    log(f"Uploading {total_vectors} vectors in batches of {BATCH_SIZE}")
    
    for i in tqdm(range(0, total_vectors, BATCH_SIZE)):
        batch_end = min(i + BATCH_SIZE, total_vectors)
        batch_size = batch_end - i
        
        batch_embeddings = embeddings[i:batch_end]
        batch_payloads = payload_list[i:batch_end]
        
        points = []
        for j in range(batch_size):
            point_id = i + j
            
            points.append(
                PointStruct(
                    id=point_id,
                    vector=batch_embeddings[j].tolist(),
                    payload=batch_payloads[j] 
                )
            )
        
        try:
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=points
            )
        except Exception as e:
            log(f"Error uploading batch {i}-{batch_end}: {e}")
            sys.exit(1)
    
    # Verify the upload
    count = client.count(COLLECTION_NAME).count
    log(f"Uploaded {count} vectors out of {total_vectors} expected")
    
    if count != total_vectors:
        log("ERROR: Not all vectors were uploaded!")
        sys.exit(1)
    
    log("Migration completed successfully!")
    
    # if VERIFY_SEARCH:
    #     log("Performing verification search to test retrieval...")
    #     # Test with a few random vectors
    #     for _ in range(3):
    #         test_idx = np.random.randint(0, total_vectors)
    #         test_vector = embeddings[test_idx]
            
    #         # Search Qdrant
    #         search_result = client.search(
    #             collection_name=COLLECTION_NAME,
    #             query_vector=test_vector.tolist(),
    #             limit=5
    #         )
            
    #         log(f"Test vector #{test_idx} results:")
    #         for i, result in enumerate(search_result):
    #             log(f"  Result #{i+1}: ID={result.id}, Distance={result.score}")
    #             if i == 0 and result.id == test_idx:
    #                 log(" First result matches input vector (as expected)")
    
    log("Migration and verification complete! Your Qdrant collection is ready.")

if __name__ == "__main__":
    main()




