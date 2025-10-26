import os
from document_processor import DocumentProcessor
from qwen_embedding import QwenEmbedding
from faiss_vectorstore import FAISSVectorStore
from qwen_llm import QwenLLM, RAGChainBuilder

class LocalKnowledgeChatbot:
    """本地知识智能客服主类"""
    
    def __init__(self, 
                 data_dir: str = "./data",
                 vector_store_dir: str = "./vector_stores",
                 api_key: str = None,
                 embedding_model: str = "text-embedding-v2",
                 llm_model: str = "qwen-max",
                 chunk_size: int = 500,
                 chunk_overlap: int = 50):
        """
        初始化客服系统
        
        Args:
            data_dir: 知识库txt文件目录
            vector_store_dir: 向量库存储目录
            api_key: DashScope API密钥，默认从环境变量获取
            embedding_model: 嵌入模型名称
            llm_model: 大语言模型名称
            chunk_size: 文档切分块大小
            chunk_overlap: 文档切分重叠大小
        """
        self.data_dir = data_dir
        self.vector_store_dir = vector_store_dir
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("DASHSCOPE_API_KEY must be provided either as argument or through environment variable.")
        
        # 初始化各组件
        self.doc_processor = DocumentProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.embedding_model = QwenEmbedding(api_key=self.api_key, model=embedding_model)
        self.vector_store = FAISSVectorStore(embeddings=self.embedding_model.embeddings, vector_store_dir=vector_store_dir)
        self.qwen_llm = QwenLLM(api_key=self.api_key, model=llm_model)
        
        # 加载或创建向量库
        self.vector_store.create_or_load_vector_store(docs_path=self.data_dir if not os.path.exists(os.path.join(vector_store_dir, "faiss_index")) else None)
        
        # 如果向量库是新建的，需要处理文档并添加
        if self.vector_store.vector_store is None and os.path.exists(self.data_dir):
            print("Processing documents and building vector store...")
            raw_docs = self.doc_processor.load_documents(self.data_dir)
            split_docs = self.doc_processor.split_documents(raw_docs)
            self.vector_store.add_documents(split_docs)
            self.vector_store.save_vector_store()
        elif self.vector_store.vector_store is None:
            raise ValueError(f"No vector store found and no documents found in {data_dir} to create one.")
        
        # 构建RAG链
        self.rag_chain_builder = RAGChainBuilder(
            llm=self.qwen_llm.get_llm(),
            vector_store=self.vector_store.vector_store
        )
        self.qa_chain = self.rag_chain_builder.build_qa_chain()
    
    def ask(self, query: str) -> dict:
        """
        回答用户问题
        
        Args:
            query: 用户问题
            
        Returns:
            包含答案和来源文档的字典
        """
        result = self.qa_chain({"question": query})  # ConversationalRetrievalChain使用"question"作为输入键
        return {
            "answer": result["answer"],
            "source_documents": result.get("source_documents", [])
        }
    
    def reset_memory(self):
        """重置对话记忆"""
        self.rag_chain_builder.memory.clear()