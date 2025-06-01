from abc import ABC, abstractmethod
from typing import List
from langchain_core.documents import Document
import numpy as np

class BaseRetriever(ABC):
    def __init__(self, embeddings_path: str, texts_path: str, k: int = 3, llm=None):
        self.k = k
        self.llm = llm
        self.setup_index(embeddings_path, texts_path)
    
    @abstractmethod
    def setup_index(self, embeddings_path: str, texts_path: str):
        pass
    
    @abstractmethod
    def get_query_embedding(self, text: str) -> np.ndarray:
        pass
    
    @abstractmethod
    def get_relevant_documents(self, query: str, use_hyde: bool = False) -> List[Document]:
        pass
    
    def set_k(self, k: int):
        self.k = k