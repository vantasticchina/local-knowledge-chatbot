from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversation.prompt import CONVERSATION_TEMPLATE
from langchain_dashscope import ChatDashScope, DashScopeGeneration
import os

class QwenLLM:
    """千问大语言模型封装类"""
    
    def __init__(self, api_key: str = None, model: str = "qwen-max"):
        """
        初始化千问LLM
        
        Args:
            api_key: DashScope API密钥，默认从环境变量获取
            model: 使用的大模型名称
        """
        api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("DASHSCOPE_API_KEY must be provided either as argument or through environment variable.")
        self.llm = ChatDashScope(
            dashscope_api_key=api_key,
            model_name=model
        )
        
    def get_llm(self):
        """获取LLM实例"""
        return self.llm

class RAGChainBuilder:
    """构建RAG链的类"""
    
    def __init__(self, llm, vector_store, memory=None):
        """
        初始化RAG链构建器
        
        Args:
            llm: 语言模型实例
            vector_store: 向量存储实例
            memory: 对话记忆实例
        """
        self.llm = llm
        self.vector_store = vector_store
        self.memory = memory or ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            input_key="query"
        )
    
    def build_qa_chain(self):
        """
        构建问答链
        
        Returns:
            配置好的QA链
        """
        # 定义提示词模板
        template = """请根据以下上下文信息回答问题。如果问题与上下文无关，请直接回答“抱歉，根据现有知识无法回答您的问题。”
        
        上下文信息:
        {context}
        
        历史对话:
        {chat_history}
        
        问题: {query}
        回答:"""
        
        QA_CHAIN_PROMPT = PromptTemplate(
            input_variables=["context", "chat_history", "query"],
            template=template
        )
        
        # 创建RAG链
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(),
            chain_type_kwargs={
                "prompt": QA_CHAIN_PROMPT,
                "memory": self.memory
            },
            return_source_documents=True
        )
        
        return qa_chain