"""
LangChain Beginner Tutorial: Key Concepts Demonstration
This application showcases essential LangChain features for beginners.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain, RetrievalQA
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.memory import ConversationBufferMemory
try:
    from langchain_community.vectorstores import Chroma
    from langchain_community.document_loaders import TextLoader
    from langchain_text_splitters import CharacterTextSplitter
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

# Load environment variables
load_dotenv()

class LangChainTutorial:
    def __init__(self):
        """Initialize the tutorial with OpenAI LLM"""
        self.llm = ChatOpenAI(
            temperature=0.7,
            model="gpt-3.5-turbo"
        )
        print("🚀 LangChain Tutorial Initialized!")
    
    def demo_basic_llm(self):
        """Demo 1: Basic LLM interaction"""
        print("\n" + "="*50)
        print("DEMO 1: Basic LLM Interaction")
        print("="*50)
        
        messages = [
            SystemMessage(content="You are a helpful coding assistant."),
            HumanMessage(content="Explain what LangChain is in one sentence.")
        ]
        
        response = self.llm(messages)
        print(f"Response: {response.content}")
    
    def demo_prompt_templates(self):
        """Demo 2: Using Prompt Templates"""
        print("\n" + "="*50)
        print("DEMO 2: Prompt Templates")
        print("="*50)
        
        template = ChatPromptTemplate.from_messages([
            ("system", "You are an expert in {topic}."),
            ("human", "Explain {concept} to a beginner in simple terms.")
        ])
        
        chain = LLMChain(llm=self.llm, prompt=template)
        
        result = chain.run(
            topic="programming",
            concept="APIs"
        )
        print(f"Explanation: {result}")
    
    def demo_memory(self):
        """Demo 3: Conversation Memory"""
        print("\n" + "="*50)
        print("DEMO 3: Conversation Memory")
        print("="*50)
        
        memory = ConversationBufferMemory()
        
        # Simulate a conversation
        memory.chat_memory.add_user_message("My name is Alice")
        memory.chat_memory.add_ai_message("Nice to meet you, Alice!")
        memory.chat_memory.add_user_message("What's my name?")
        
        print("Conversation History:")
        for message in memory.chat_memory.messages:
            role = "Human" if hasattr(message, 'content') and message.__class__.__name__ == "HumanMessage" else "AI"
            print(f"{role}: {message.content}")
    
    def demo_tools_and_agents(self):
        """Demo 4: Tools and Agents"""
        print("\n" + "="*50)
        print("DEMO 4: Tools and Agents")
        print("="*50)
        
        # Define custom tools
        def calculator(expression: str) -> str:
            """Calculate mathematical expressions"""
            try:
                result = eval(expression)
                return f"The result is: {result}"
            except:
                return "Invalid mathematical expression"
        
        def word_counter(text: str) -> str:
            """Count words in text"""
            word_count = len(text.split())
            return f"Word count: {word_count}"
        
        tools = [
            Tool(
                name="Calculator",
                func=calculator,
                description="Use this to perform mathematical calculations"
            ),
            Tool(
                name="WordCounter",
                func=word_counter,
                description="Use this to count words in text"
            )
        ]
        
        # Initialize agent
        agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )
        
        # Test the agent
        print("Agent solving: 'Calculate 15 * 8 + 12'")
        result = agent.run("Calculate 15 * 8 + 12")
        print(f"Agent Result: {result}")

        print("Agent solving: 'Count words in this sentence'")
        word_count = agent.run("Count words in this sentence")
        print(f"Agent Result: {word_count}")
    
    def demo_rag(self):
        """Demo 5: RAG (Retrieval-Augmented Generation)"""
        print("\n" + "="*50)
        print("DEMO 5: RAG (Retrieval-Augmented Generation)")
        print("="*50)
        
        if not CHROMA_AVAILABLE:
            print("❌ ChromaDB not installed. To enable RAG demo:")
            print("   Windows: python -m pip install chromadb")
            print("   Or: py -m pip install chromadb")
            print("   Then run: pip install langchain-text-splitters")
            return
        
        try:
            # Load and split documents
            loader = TextLoader("sample_docs.txt")
            documents = loader.load()
            
            text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            texts = text_splitter.split_documents(documents)
            
            # Create embeddings and vector store
            embeddings = OpenAIEmbeddings()
            vectorstore = Chroma.from_documents(texts, embeddings)
            
            # Create RAG chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(),
                verbose=False
            )
            
            # Test RAG
            query = "What are the key components of LangChain?"
            print(f"Query: {query}")
            result = qa_chain.run(query)
            print(f"RAG Answer: {result}")
            
            query2 = "How does RAG work?"
            print(f"\nQuery: {query2}")
            result2 = qa_chain.run(query2)
            print(f"RAG Answer: {result2}")
            
        except Exception as e:
            print(f"❌ RAG Demo Error: {e}")
            print("Make sure sample_docs.txt exists in the current directory")
    
    def run_all_demos(self):
        """Run all demonstration examples"""
        print("🎓 Welcome to LangChain Beginner Tutorial!")
        print("This tutorial covers 5 key LangChain concepts:")
        print("1. Basic LLM Interaction")
        print("2. Prompt Templates")
        print("3. Conversation Memory")
        print("4. Tools and Agents")
        print("5. RAG (Retrieval-Augmented Generation)")
        
        try:
            self.demo_basic_llm()
            self.demo_prompt_templates()
            self.demo_memory()
            self.demo_tools_and_agents()
            if CHROMA_AVAILABLE:
                self.demo_rag()
            else:
                print("\n⚠️  RAG demo skipped - ChromaDB not installed")
            
            print("\n" + "="*50)
            print("🎉 Tutorial Complete!")
            print("Key Takeaways:")
            print("• LLMs are the foundation of LangChain")
            print("• Prompt templates make interactions reusable")
            print("• Memory enables contextual conversations")
            print("• Agents can use tools to solve complex tasks")
            print("• RAG combines LLMs with external knowledge")
            print("="*50)
            
        except Exception as e:
            print(f"❌ Error: {e}")
            print("Make sure to set your OPENAI_API_KEY in .env file")

if __name__ == "__main__":
    tutorial = LangChainTutorial()
    tutorial.run_all_demos()