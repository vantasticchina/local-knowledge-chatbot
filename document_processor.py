import os
from typing import List
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DocumentProcessor:
    """处理文档加载和切分的类"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        初始化文档处理器
        
        Args:
            chunk_size: 文本块大小
            chunk_overlap: 块重叠大小，保证上下文连续性
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

    def load_documents(self, data_dir: str) -> List:
        """
        从指定目录加载所有txt文件
        
        Args:
            data_dir: 包含txt文件的目录路径
            
        Returns:
            加载的文档列表
        """
        documents = []
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.lower().endswith('.txt'):
                    file_path = os.path.join(root, file)
                    loader = TextLoader(file_path, encoding='utf-8')
                    documents.extend(loader.load())
        return documents

    def split_documents(self, documents: List) -> List:
        """
        切分文档
        
        Args:
            documents: 原始文档列表
            
        Returns:
            切分后的文档块列表
        """
        return self.text_splitter.split_documents(documents)