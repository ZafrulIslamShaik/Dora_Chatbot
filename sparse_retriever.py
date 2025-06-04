"""
Sparse Retriever - Modified to use Qdrant

Implements BM25-based sparse retrieval using documents from Qdrant vector database.

Input: Qdrant collection (dora_embeddings)
Output: Ranked relevant documents based on BM25 scoring
"""

import json
from llama_index.core.schema import TextNode, Document
from llama_index.retrievers.bm25 import BM25Retriever
from qdrant_client import QdrantClient

class SparseRetriever:
    def __init__(self, sparse_k: int = 5):
        self.sparse_k = sparse_k
        
        # Connect to Qdrant
        print(f"Connecting to Qdrant for sparse retrieval...")
        self.qdrant_client = QdrantClient(host="localhost", port=6333)
        self.collection_name = "dora_embeddings"
        
        # Load all documents from Qdrant collection
        print(f"Loading documents from Qdrant collection '{self.collection_name}'...")
        all_points = self.qdrant_client.scroll(
            collection_name=self.collection_name,
            limit=10000,  
            with_payload=True
        )[0]
        
        print(f"Loaded {len(all_points)} documents from Qdrant")
        
        # Build TextNodes from Qdrant data
        nodes = []
        for point in all_points:
            text_content = point.payload.get("text", "")
            
            # Handle text format (could be string or list)
            if isinstance(text_content, list):
                text_content = text_content[0] if text_content else ""
            
            metadata = {k: v for k, v in point.payload.items() if k != "text"}
            metadata["qdrant_id"] = point.id 
            
            nodes.append(TextNode(text=text_content, metadata=metadata))
        
        # Create BM25 index from Qdrant documents
        self.bm25_retriever = BM25Retriever.from_defaults(
            nodes=nodes,
            similarity_top_k=self.sparse_k
        )
        
        print(f"BM25 index created with {len(nodes)} documents")

    def get_relevant_documents(self, query: str):
        print(f"\n" + "="*80)
        print("SPARSE RETRIEVER DEBUG (QDRANT VERSION)")
        print("="*80)
        print(f"Query: {query}")
        print("-"*80)
        
        bm25_results = self.bm25_retriever.retrieve(query)
        
        print(f"BM25 RESULTS ({len(bm25_results)}):")
        for i, result in enumerate(bm25_results):
            text_preview = result.text[:100]
            score = getattr(result, 'score', 'NO_SCORE')
            qdrant_id = result.metadata.get("qdrant_id", "unknown")
            print(f"  {i+1}. Score: {score} | Qdrant ID: {qdrant_id} | {text_preview}...")

        documents = []
        for result in bm25_results[:self.sparse_k]:
            doc = Document(
                text=result.text,
                metadata={
                    **result.metadata,
    
                    "retriever_type": "sparse_qdrant"
                }
            )
            documents.append(doc)
            
        print(f"Returning {len(documents)} documents")
        print("="*80)
        return documents










