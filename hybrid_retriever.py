"""
Hybrid Retrieval Module

Input: Qdrant vector database (both dense and sparse)
Output: Fused and reranked relevant documents

"""

import json
import logging
from typing import List
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.schema import TextNode, Document
from llama_index.core.embeddings import BaseEmbedding
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from document_embeddings.embedding_generator import get_azure_embedding
from EVALUATION.config import *

# === Configuration ===
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "dora_embeddings"
TEXTS_FILE = "document_embeddings/split_documents.json"
ALPHA = 0.7
TOP_K = 10


class HybridRetriever:
    
    def __init__(self, alpha: float = ALPHA, top_k: int = TOP_K):
        """
        Initialize hybrid retriever with sparse and dense components.
        
        Args:
            alpha (float): Weight for dense retrieval (1-alpha for sparse)
            top_k (int): Number of documents to retrieve from each method
        """
        self.alpha = alpha
        self.top_k = top_k

        logging.info(f"Initializing HybridRetriever (alpha={alpha}, top_k={top_k})")

        # Connect to Qdrant (used for both sparse and dense)
        qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        
        # Load documents from Qdrant collection for BM25 indexing
        all_points = qdrant_client.scroll(
            collection_name=COLLECTION_NAME,
            limit=10000,
            with_payload=True
        )[0]
        
        print(f"Loaded {len(all_points)} documents from Qdrant for BM25")
        
        # Build TextNodes from Qdrant data
        self.nodes = []
        for point in all_points:
            text_content = point.payload.get("text", "")
            
            if isinstance(text_content, list):
                text_content = text_content[0] if text_content else ""
            
            metadata = {k: v for k, v in point.payload.items() if k != "text"}
            metadata["qdrant_id"] = point.id
            
            self.nodes.append(TextNode(text=text_content, metadata=metadata))
        
        logging.info(f"Loaded {len(self.nodes)} TextNodes for BM25 from Qdrant")

        # Initialize BM25 sparse retriever
        self.sparse_retriever = BM25Retriever.from_defaults(
            nodes=self.nodes,
            similarity_top_k=self.top_k
        )

        # Configure LlamaIndex settings for Azure embeddings
        Settings.embed_model = AzureEmbedding()
        Settings.llm = None 

        # Initialize Qdrant vector store for dense retrieval
        vector_store = QdrantVectorStore(client=qdrant_client, collection_name=COLLECTION_NAME)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Build vector index from existing Qdrant collection
        vector_index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_context
        )
        self.dense_retriever = vector_index.as_retriever(similarity_top_k=self.top_k)

        # Create fusion retriever with reciprocal rank fusion
        self.retriever = QueryFusionRetriever(
            retrievers=[self.dense_retriever, self.sparse_retriever],
            retriever_weights=[self.alpha, 1-self.alpha],
            similarity_top_k=self.top_k,
            mode="reciprocal_rerank",
            num_queries=1,  
            llm=None, 
            use_async=False  
        )

    def get_relevant_documents(self, query: str) -> List[Document]:
        """
        Retrieve documents using hybrid approach with reciprocal rank fusion.

        """
        results = self.retriever.retrieve(query)
        
        result_docs = [
            Document(text=r.node.text, metadata=r.node.metadata)
            for r in results
        ]
        return result_docs
        
class AzureEmbedding(BaseEmbedding):
    """
    Custom LlamaIndex embedding model wrapper for Azure OpenAI embeddings.
    
    Provides standardized interface for both synchronous and asynchronous
    embedding generation using Azure OpenAI embedding models.
    """
    
    def _get_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for query text."""
        return get_azure_embedding(query)
    
    def _get_text_embedding(self, text: str) -> List[float]:
        """Generate embedding for document text."""
        return get_azure_embedding(text)
    
    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Asynchronous query embedding generation."""
        return self._get_query_embedding(query)
    
    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Asynchronous text embedding generation."""
        return self._get_text_embedding(text)