import os
from rag_system import LocalKnowledgeChatbot

def main():
    """主函数，启动交互式对话"""
    print("Initializing Local Knowledge Chatbot...")
    
    # 创建客服实例
    chatbot = LocalKnowledgeChatbot(
        data_dir="./data",  # 确保此目录包含txt文件
        vector_store_dir="./vector_stores",
        # api_key=\"your-api-key-here\"  # 可以直接传入，否则从环境变量加载
    )
    
    print("Chatbot initialized successfully!")
    print("You can start chatting. Type 'exit' to quit, 'reset' to clear conversation history.\\n")
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        elif user_input.lower() == 'reset':
            chatbot.reset_memory()
            print("Chatbot: Conversation history has been cleared.")
            continue
        
        try:
            response = chatbot.ask(user_input)
            print(f"Chatbot: {response['answer']}\\n")
            
            # 如果需要查看来源文档，取消下面的注释
            # print(\"Source documents:\")
            # for doc in response['source_documents']:
            #     print(f\"- {doc.metadata.get('source', 'Unknown source')}: {doc.page_content[:100]}...\")
            # print()
            
        except Exception as e:
            print(f"Chatbot: An error occurred: {str(e)}\\n")

if __name__ == "__main__":
    main()