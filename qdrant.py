# import os
# import sys
# import numpy as np
# import json
# import time
# from tqdm import tqdm
# from qdrant_client import QdrantClient
# from qdrant_client.http.models import Distance, VectorParams, PointStruct, OptimizersConfigDiff
# from qdrant_client.http.exceptions import UnexpectedResponse

# # === CONFIGURATION ===
# QDRANT_HOST = "localhost"
# QDRANT_PORT = 6333

# COLLECTION_NAME = "dora_embeddings"
# EMBEDDING_FILE = "document_embeddings/split_documents.npy"
# CHUNKS_FILE = "document_embeddings/split_documents.json"  

# BATCH_SIZE = 100

# def log(message):
#     """Helper function for logging with timestamp"""
#     timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
#     print(f"[{timestamp}] {message}")

# def main():
#     log("Starting migration from FAISS to Qdrant")

#     # Connect to Qdrant
#     log(f"Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
#     client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    
#     try:
#         collection_info = client.get_collection(COLLECTION_NAME)
#         log(f"Collection '{COLLECTION_NAME}' already exists. Recreating...")
#         client.delete_collection(COLLECTION_NAME)
#     except UnexpectedResponse:
#         log(f"Collection '{COLLECTION_NAME}' does not exist. Creating new.")

#     # Load embeddings
#     log(f"Loading embeddings from {EMBEDDING_FILE}")
#     embeddings = np.load(EMBEDDING_FILE)
#     vector_size = embeddings.shape[1]
#     total_vectors = embeddings.shape[0]
    
#     with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
#         chunks = json.load(f)
    
#     payload_list = []
#     for chunk in chunks:
#         text_content = ""
#         if "text" in chunk:
#             if isinstance(chunk["text"], list) and chunk["text"]:
#                 text_content = chunk["text"][0] 
#             else:
#                 text_content = chunk.get("text", "")
                
#         # Create payload with text and metadata
#         payload = {
#             "text": text_content,
#             **chunk.get("metadata", {})  # Include all metadata fields
#         }
#         payload_list.append(payload)
    
#     # Validate chunk count
#     if len(payload_list) != total_vectors:
#         log(f"ERROR: Mismatch between embeddings ({total_vectors}) and chunks ({len(payload_list)})!")
#         sys.exit(1)
    
#     # Create collection with L2 distance (Euclidean) to match FAISS IndexFlatL2
#     log(f"Creating collection '{COLLECTION_NAME}' with vector size {vector_size}")
#     client.create_collection(
#         collection_name=COLLECTION_NAME,
#         vectors_config=VectorParams(
#             size=vector_size,
#             distance=Distance.EUCLID  
#         ),
#         optimizers_config=OptimizersConfigDiff(
#             indexing_threshold=20000  
#         )
#     )
    
#     # Upload vectors in batches
#     log(f"Uploading {total_vectors} vectors in batches of {BATCH_SIZE}")
    
#     for i in tqdm(range(0, total_vectors, BATCH_SIZE)):
#         batch_end = min(i + BATCH_SIZE, total_vectors)
#         batch_size = batch_end - i
        
#         batch_embeddings = embeddings[i:batch_end]
#         batch_payloads = payload_list[i:batch_end]
        
#         points = []
#         for j in range(batch_size):
#             point_id = i + j
            
#             points.append(
#                 PointStruct(
#                     id=point_id,
#                     vector=batch_embeddings[j].tolist(),
#                     payload=batch_payloads[j] 
#                 )
#             )
        
#         try:
#             client.upsert(
#                 collection_name=COLLECTION_NAME,
#                 points=points
#             )
#         except Exception as e:
#             log(f"Error uploading batch {i}-{batch_end}: {e}")
#             sys.exit(1)
    
#     # Verify the upload
#     count = client.count(COLLECTION_NAME).count
#     log(f"Uploaded {count} vectors out of {total_vectors} expected")
    
#     if count != total_vectors:
#         log("ERROR: Not all vectors were uploaded!")
#         sys.exit(1)
    
#     log("Migration completed successfully!")
    
#     log("Migration and verification complete! Your Qdrant collection is ready.")

# if __name__ == "__main__":
#     main()




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

# Main DORA collection configuration
DORA_COLLECTION_NAME = "dora_embeddings"
DORA_EMBEDDING_FILE = "document_embeddings/small.npy"
DORA_CHUNKS_FILE = "document_embeddings/small.json"

# Cross-references collection configuration
CROSS_REF_COLLECTION_NAME = "cross_references"
CROSS_REF_EMBEDDING_FILE = "document_embeddings/cross_references.npy"
CROSS_REF_CHUNKS_FILE = "document_embeddings/cross_references.json"

BATCH_SIZE = 100

def log(message):
    """Helper function for logging with timestamp"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def upload_collection(client, collection_name, embedding_file, chunks_file, is_cross_ref=False):
    """Upload a collection to Qdrant"""
    
    try:
        collection_info = client.get_collection(collection_name)
        log(f"Collection '{collection_name}' already exists. Recreating...")
        client.delete_collection(collection_name)
    except UnexpectedResponse:
        log(f"Collection '{collection_name}' does not exist. Creating new.")

    # Load embeddings
    log(f"Loading embeddings from {embedding_file}")
    embeddings = np.load(embedding_file)
    vector_size = embeddings.shape[1]
    total_vectors = embeddings.shape[0]
    
    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    payload_list = []
    
    if is_cross_ref:
        # Handle cross-references JSON format
        for cross_ref in chunks:
            text_content = cross_ref.get("corresponding_text", "")
            reference = cross_ref.get("reference", "")
            
            payload = {
                "text": text_content,
                "reference": reference,
                "document_type": "cross_reference"
            }
            payload_list.append(payload)
    else:
        # Handle main DORA collection format
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
        log(f"ERROR: Mismatch between embeddings ({total_vectors}) and chunks ({len(payload_list)}) for {collection_name}!")
        return False
    
    # Create collection with L2 distance (Euclidean) to match FAISS IndexFlatL2
    log(f"Creating collection '{collection_name}' with vector size {vector_size}")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE    
        ),
        optimizers_config=OptimizersConfigDiff(
            indexing_threshold=20000  
        )
    )
    
    # Upload vectors in batches
    log(f"Uploading {total_vectors} vectors in batches of {BATCH_SIZE}")
    
    for i in tqdm(range(0, total_vectors, BATCH_SIZE), desc=f"Uploading {collection_name}"):
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
                collection_name=collection_name,
                points=points
            )
        except Exception as e:
            log(f"Error uploading batch {i}-{batch_end} for {collection_name}: {e}")
            return False
    
    # Verify the upload
    count = client.count(collection_name).count
    log(f"Uploaded {count} vectors out of {total_vectors} expected for {collection_name}")
    
    if count != total_vectors:
        log(f"ERROR: Not all vectors were uploaded for {collection_name}!")
        return False
    
    log(f"Collection '{collection_name}' uploaded successfully!")
    return True

def main():
    log("Starting upload of BOTH collections to Qdrant")

    # Connect to Qdrant
    log(f"Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    
    # Upload main DORA collection
    log("=" * 60)
    log("UPLOADING MAIN DORA COLLECTION")
    log("=" * 60)
    
    success_dora = upload_collection(
        client, 
        DORA_COLLECTION_NAME, 
        DORA_EMBEDDING_FILE, 
        DORA_CHUNKS_FILE, 
        is_cross_ref=False
    )
    
    if not success_dora:
        log("Failed to upload DORA collection. Stopping.")
        sys.exit(1)
    
    # Upload cross-references collection
    log("\n" + "=" * 60)
    log("UPLOADING CROSS-REFERENCES COLLECTION")
    log("=" * 60)
    
    success_cross_ref = upload_collection(
        client, 
        CROSS_REF_COLLECTION_NAME, 
        CROSS_REF_EMBEDDING_FILE, 
        CROSS_REF_CHUNKS_FILE, 
        is_cross_ref=True
    )
    
    if not success_cross_ref:
        log("Failed to upload cross-references collection. Stopping.")
        sys.exit(1)
    
    # Final verification
    log("\n" + "=" * 60)
    log("FINAL VERIFICATION")
    log("=" * 60)
    
    dora_count = client.count(DORA_COLLECTION_NAME).count
    cross_ref_count = client.count(CROSS_REF_COLLECTION_NAME).count
    
    log(f"✅ DORA collection: {dora_count} vectors")
    log(f"✅ Cross-references collection: {cross_ref_count} vectors")
    log(f"✅ Total vectors uploaded: {dora_count + cross_ref_count}")
    
    log("\n🎉 BOTH COLLECTIONS UPLOADED SUCCESSFULLY!")
    log("Your Qdrant database is ready for enhanced RAG with cross-references!")

if __name__ == "__main__":
    main()

