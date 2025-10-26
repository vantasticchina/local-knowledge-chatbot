import os
from typing import List, Optional
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

class FAISSVectorStore:
    """FAISS向量存储和检索类"""
    
    def __init__(self, embeddings: Embeddings, vector_store_dir: str = "./vector_stores"):
        """
        初始化FAISS向量存储
        
        Args:
            embeddings: 嵌入模型实例
            vector_store_dir: 向量库保存目录
        """
        self.embeddings = embeddings
        self.vector_store_dir = vector_store_dir
        self.vector_store: Optional[FAISS] = None
        os.makedirs(vector_store_dir, exist_ok=True)
    
    def create_or_load_vector_store(self, docs_path: str = None) -> 'FAISSVectorStore':
        """
        创建或加载向量库
        
        Args:
            docs_path: 文档路径，如果向量库不存在且提供了此路径，则创建向量库
            
        Returns:
            自身实例
        """
        # 尝试加载已存在的向量库
        vector_store_path = os.path.join(self.vector_store_dir, "faiss_index")
        if os.path.exists(vector_store_path):
            print(f"Loading existing vector store from {vector_store_path}")
            self.vector_store = FAISS.load_local(
                vector_store_path, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
        elif docs_path:
            print("No existing vector store found, creating new one...")
            # 此处需要文档processor来处理文档，将在rag_system中集成
            # 暂时返回实例，实际创建在后续方法中
        else:
            raise ValueError("No existing vector store found and no docs_path provided to create one.")
        
        return self
    
    def add_documents(self, documents: List):
        """
        添加文档到向量库
        
        Args:
            documents: 要添加的文档列表
        """
        if self.vector_store is None:
            # 首次创建
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
        else:
            # 追加到现有向量库
            self.vector_store.add_documents(documents)
    
    def save_vector_store(self):
        """保存向量库到磁盘"""
        if self.vector_store:
            vector_store_path = os.path.join(self.vector_store_dir, "faiss_index")
            self.vector_store.save_local(vector_store_path)
            print(f"Vector store saved to {vector_store_path}")
    
    def similarity_search(self, query: str, k: int = 4) -> List:
        """
        相似性搜索
        
        Args:
            query: 查询文本
            k: 返回结果数量
            
        Returns:
            相似文档列表
        """
        if self.vector_store is None:
            raise ValueError("Vector store is not initialized.")
        return self.vector_store.similarity_search(query, k=k)
    
    def similarity_search_with_score(self, query: str, k: int = 4) -> List:
        """
        相似性搜索，返回相似度分数
        
        Args:
            query: 查询文本
            k: 返回结果数量
            
        Returns:
            (文档, 相似度分数) 元组列表
        """
        if self.vector_store is None:
            raise ValueError("Vector store is not initialized.")
        return self.vector_store.similarity_search_with_score(query, k=k)