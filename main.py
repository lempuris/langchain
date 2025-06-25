"""
LangChain Beginner Tutorial: Key Concepts Demonstration
This application showcases essential LangChain features for beginners.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.memory import ConversationBufferMemory

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
    
    def run_all_demos(self):
        """Run all demonstration examples"""
        print("🎓 Welcome to LangChain Beginner Tutorial!")
        print("This tutorial covers 4 key LangChain concepts:")
        print("1. Basic LLM Interaction")
        print("2. Prompt Templates")
        print("3. Conversation Memory")
        print("4. Tools and Agents")
        
        try:
            self.demo_basic_llm()
            self.demo_prompt_templates()
            self.demo_memory()
            self.demo_tools_and_agents()
            
            print("\n" + "="*50)
            print("🎉 Tutorial Complete!")
            print("Key Takeaways:")
            print("• LLMs are the foundation of LangChain")
            print("• Prompt templates make interactions reusable")
            print("• Memory enables contextual conversations")
            print("• Agents can use tools to solve complex tasks")
            print("="*50)
            
        except Exception as e:
            print(f"❌ Error: {e}")
            print("Make sure to set your OPENAI_API_KEY in .env file")

if __name__ == "__main__":
    tutorial = LangChainTutorial()
    tutorial.run_all_demos()