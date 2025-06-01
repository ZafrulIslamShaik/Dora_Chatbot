"""
Sparse Retriever 

Implements BM25-based sparse retrieval using LlamaIndex for document search and ranking.

Input: split_documents.json (chunked documents)
Output: Ranked relevant documents based on BM25 scoring
"""

import json
from llama_index.core.schema import TextNode, Document
from llama_index.retrievers.bm25 import BM25Retriever

class SparseRetriever:
    def __init__(self, texts_path: str, sparse_k: int = 5):
        self.sparse_k = sparse_k
        self.texts = self._load_texts(texts_path)
        
        nodes = []
        for chunk in self.texts:
            text_content = chunk["text"]
            if isinstance(text_content, list):
                text_content = text_content[0] if text_content else ""
            
            nodes.append(TextNode(text=text_content, metadata=chunk["metadata"]))
        
        self.bm25_retriever = BM25Retriever.from_defaults(
            nodes=nodes,
            similarity_top_k=self.sparse_k
        )

    def _load_texts(self, texts_path):
        with open(texts_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        return [chunk for chunk in chunks if "text" in chunk and "metadata" in chunk]

    def get_relevant_documents(self, query: str):
        bm25_results = self.bm25_retriever.retrieve(query)

        documents = []
        for result in bm25_results[:self.sparse_k]:
            doc = Document(
                text=result.text,
                metadata=result.metadata
            )
            documents.append(doc)
            
        return documents






















# from llama_index.retrievers.bm25 import BM25Retriever
# from llama_index.core.schema import TextNode
# import json
# import logging
# from langchain_core.documents import Document

# class SparseRetriever:
#     def __init__(self, texts_path: str, sparse_k: int = 5):
#         self.sparse_k = sparse_k
#         self.texts = self._load_texts(texts_path)
        
#         # Create nodes with proper text handling
#         nodes = []
#         for chunk in self.texts:
#             # Handle case where text is a list
#             text_content = chunk["text"]
#             if isinstance(text_content, list):
#                 text_content = text_content[0] if text_content else ""
            
#             nodes.append(TextNode(text=text_content, metadata=chunk["metadata"]))
        
#         self.bm25_retriever = BM25Retriever.from_defaults(
#             nodes=nodes,
#             similarity_top_k=self.sparse_k
#         )

#     def _load_texts(self, texts_path):
#         with open(texts_path, "r", encoding="utf-8") as f:
#             chunks = json.load(f)
#         return [chunk for chunk in chunks if "text" in chunk and "metadata" in chunk]

#     def get_relevant_documents(self, query: str):
#         bm25_results = self.bm25_retriever.retrieve(query)

#         return [Document(page_content=result.text, metadata=result.metadata) for result in bm25_results[:self.sparse_k]]


















# from llama_index.retrievers.bm25 import BM25Retriever
# from llama_index.core.schema import TextNode, Document
# import json
# import logging

# class SparseRetriever:
#     def __init__(self, texts_path: str, sparse_k: int = 5):
#         self.sparse_k = sparse_k
#         self.texts = self._load_texts(texts_path)
        
#         # Create nodes with proper text handling
#         nodes = []
#         for chunk in self.texts:
#             # Handle case where text is a list
#             text_content = chunk["text"]
#             if isinstance(text_content, list):
#                 text_content = text_content[0] if text_content else ""
            
#             nodes.append(TextNode(text=text_content, metadata=chunk["metadata"]))
        
#         self.bm25_retriever = BM25Retriever.from_defaults(
#             nodes=nodes,
#             similarity_top_k=self.sparse_k
#         )

#     def _load_texts(self, texts_path):
#         with open(texts_path, "r", encoding="utf-8") as f:
#             chunks = json.load(f)
#         return [chunk for chunk in chunks if "text" in chunk and "metadata" in chunk]

#     def get_relevant_documents(self, query: str):
#         bm25_results = self.bm25_retriever.retrieve(query)

#         # Convert LlamaIndex nodes to Documents
#         documents = []
#         for result in bm25_results[:self.sparse_k]:
#             doc = Document(
#                 text=result.text,
#                 metadata=result.metadata
#             )
#             documents.append(doc)
            
#         return documents










