from typing import List
from langchain_core.embeddings import Embeddings
from langchain_community.embeddings import DashScopeEmbeddings
import os

class QwenEmbedding(Embeddings):
    """千问Embedding模型封装类"""
    
    def __init__(self, api_key: str = None, model: str = "text-embedding-v2"):
        """
        初始化千问Embedding
        
        Args:
            api_key: DashScope API密钥，默认从环境变量获取
            model: 使用的嵌入模型名称
        """
        api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("DASHSCOPE_API_KEY must be provided either as argument or through environment variable.")
        self.embeddings = DashScopeEmbeddings(
            dashscope_api_key=api_key,
            model=model
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        对文档列表进行嵌入
        
        Args:
            texts: 文档文本列表
            
        Returns:
            嵌入向量列表
        """
        return self.embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        """
        对单个查询进行嵌入
        
        Args:
            text: 查询文本
            
        Returns:
            嵌入向量
        """
        return self.embeddings.embed_query(text)