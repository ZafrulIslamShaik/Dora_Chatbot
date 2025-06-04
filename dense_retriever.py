
"""
Dense Retriever Module

Implements vector-based dense retrieval using Qdrant vector database.
Supports HyDE (Hypothetical Document Embeddings) for enhanced query understanding.

Uses: Qdrant vector database

"""

from typing import List
import numpy as np
from qdrant_client import QdrantClient
from document_embeddings.embedding_generator import get_azure_embedding
from hyde import get_dora_hyde_answer
from retriever_base import BaseRetriever
from datetime import datetime
from EVALUATION.config import *
from llama_index.core.schema import Document

class DenseRetriever(BaseRetriever):
    """
    Dense retrieval system using vector similarity search with Qdrant.
    
    Implements semantic search through embedding-based similarity matching,
    with optional HyDE enhancement for improved query representation.
    """
    
    def __init__(self, k: int = 5, llm=None):
        """
        Initialize dense retriever with Qdrant connection only.
        
        Args:
            k (int): Number of documents to retrieve (default: 5)
            llm: Language model for HyDE functionality
        """
        self.k = k
        self.llm = llm
        self.setup_index()
    
    def setup_index(self):
        """
        Set up Qdrant vector database connection.
        All text and metadata comes from Qdrant payloads.
        """
        # Initialize Qdrant vector database connection
        self.qdrant_client = QdrantClient(host="localhost", port=6333)
        
        self.collection_name = "dora_embeddings"

        try:
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            print(f"Connected to Qdrant collection '{self.collection_name}' with {collection_info.vectors_count} vectors")
        except Exception as e:
            print(f"Error connecting to Qdrant: {e}")
            raise
    
    def get_query_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding vector for query text using Azure OpenAI.
        
        Args:
            text (str): Query text to embed
            
        Returns:
            np.ndarray: Query embedding vector
        """
        embedding = get_azure_embedding(text)
        return np.array(embedding)
    
    def get_relevant_documents(self, query: str, use_hyde: bool = False) -> List[Document]:
        """
        Retrieve most relevant documents using vector similarity search.
        
        Args:
            query (str): User query for document retrieval
            use_hyde (bool): Whether to use HyDE for query enhancement
            
        Returns:
            List[Document]: Top-k most similar documents with metadata
        """
        start_time = datetime.now()
        
        # Apply HyDE enhancement if enabled and LLM available
        if use_hyde and self.llm:
            hyde_answer = get_dora_hyde_answer(self.llm, query)
            embed_start = datetime.now()
            query_embedding = self.get_query_embedding(hyde_answer)
            embed_time = (datetime.now() - embed_start).total_seconds()
            print(f"[DenseRetriever] HYDE embedding took {embed_time:.2f}s")
        else:
            embed_start = datetime.now()
            query_embedding = self.get_query_embedding(query)
            embed_time = (datetime.now() - embed_start).total_seconds()
        
        search_start = datetime.now()
        
        # Perform vector similarity search in Qdrant
        search_results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=self.k
        )
        
        search_time = (datetime.now() - search_start).total_seconds()
        
        # Build Document objects using ONLY Qdrant payload data
        documents = []
        for result in search_results:
            point_id = result.id
            payload = result.payload
            
            # Get text content directly from Qdrant payload
            text_content = payload.get("text", "")
            
            doc = Document(
                text=text_content,
                metadata={
                    **payload,
                    "embedding_index": int(point_id),
                    "score": float(result.score)
                }
            )
            
            documents.append(doc)
        
        return documents


























