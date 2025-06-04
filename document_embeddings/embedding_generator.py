"""
Embedding Generation Module

Generates vector embeddings for document chunks using Azure OpenAI embedding models.

Input: split_documents.json (chunked documents)
Output: embeddings.npy file
"""

import os
import json
import numpy as np
import requests
import dotenv
import time  



CHUNKS_FILE= "split_documents.json"
dotenv.load_dotenv(".env.local")

# Configuration for embedding storage
VECTOR_EMBEDDINGS_FOLDER = "embeddings"
os.makedirs(VECTOR_EMBEDDINGS_FOLDER, exist_ok=True)

def load_chunks() -> list:
    """Load JSON chunks and ensure they contain 'text' and 'metadata'."""
    if not os.path.exists(CHUNKS_FILE):
        raise FileNotFoundError(f"Chunks file not found: {CHUNKS_FILE}")
    
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    formatted_chunks = []
    for chunk in chunks:
        if isinstance(chunk, dict) and "text" in chunk and "metadata" in chunk:
            formatted_chunks.append(chunk)
        else:
            print(f"Skipping invalid chunk: {chunk}")

    return formatted_chunks

def get_azure_embedding(text: str) -> list:
    """
    Fetches an embedding from an Azure OpenAI embedding model.
    With error handling to prevent None returns.
    """
    try:
        # Retrieve Azure OpenAI configuration from environment
        AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
        AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
        DEPLOYMENT_NAME = os.getenv("EMBEDDING_DEPLOYMENT_NAME")

        headers = {
            "Content-Type": "application/json",
            "api-key": AZURE_OPENAI_KEY
        }

        # Construct Azure OpenAI embeddings endpoint URL
        url = f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{DEPLOYMENT_NAME}/embeddings?api-version=2023-05-15"

        payload = {"input": text}

        response = requests.post(url, headers=headers, json=payload)

        if response.status_code == 200:
            embedding = response.json()['data'][0]['embedding']
            if embedding is None or len(embedding) == 0:
                print(f"Azure returned empty embedding for text: {text[:50]}...")
                return [0.0] * 1536  
            return embedding
        else:
            print(f"Azure API error {response.status_code}: {response.text}")
            # Implement exponential backoff for rate limiting
            if response.status_code == 429:
                print("Rate limited by Azure API, waiting 10 seconds before retry...")
                time.sleep(10)
                return get_azure_embedding(text) 

            return [0.0] * 1536
    except Exception as e:
        print(f"Exception in get_azure_embedding: {e}")
        return [0.0] * 1536  

def create_and_save_embeddings():
    """Generate embeddings and save them along with metadata."""
    # Define storage path for embeddings
    embeddings_path = os.path.join(VECTOR_EMBEDDINGS_FOLDER, "embeddings.npy")
    
    # Skip generation if embeddings already exist
    if os.path.exists(embeddings_path):
        print("Existing embeddings found. Skipping generation.")
        return

    # Load and validate input chunks
    chunks = load_chunks()
    
    # Initialize storage for embeddings
    embeddings = []
    
    # Process chunks in batches to respect API rate limits
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        print(f"Processing batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}")
        batch_chunks = chunks[i:i+batch_size]
        
        for chunk in batch_chunks:
            text = chunk["text"][0].strip() if isinstance(chunk["text"], list) else chunk["text"].strip()
            metadata = chunk["metadata"]

            if text:  
                try:
                    embedding = get_azure_embedding(text)
                    embeddings.append(embedding)

                except Exception as e:
                    print(f"Failed to embed chunk {metadata.get('ID', '')}: {e}")
        
        # Implement rate limiting between batches to avoid API throttling
        if i + batch_size < len(chunks):
            print(f"Waiting 1 minute before processing the next batch...")
            time.sleep(60)  # Sleep for 60 seconds (1 minute)

    # Save embeddings
    embeddings_array = np.array(embeddings)
    np.save(embeddings_path, embeddings_array)

    print(f"Generated {len(embeddings)} embeddings and saved them successfully in {VECTOR_EMBEDDINGS_FOLDER}!")

if __name__ == "__main__":
    create_and_save_embeddings()