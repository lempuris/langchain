# -*- coding: utf-8 -*-
"""
LangGraph Beginner Tutorial: Building Stateful AI Applications
This tutorial demonstrates key LangGraph concepts for building stateful AI applications.
LangGraph is LangChain's framework for building stateful, multi-actor applications with LLMs.
"""

import os
from typing import Dict, List, Any, TypedDict, Annotated
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.runnables import RunnablePassthrough

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langgraph.graph import START

# Load environment variables
load_dotenv()

# Configure LangSmith tracing (if available)
if os.getenv("LANGCHAIN_TRACING_V2"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "langraph-tutorial")

class LangGraphTutorial:
    def __init__(self):
        """Initialize the tutorial with OpenAI LLM"""
        self.llm = ChatOpenAI(
            temperature=0.7,
            model="gpt-3.5-turbo"
        )
        print("🚀 LangGraph Tutorial Initialized!")
        print("📊 LangSmith tracing enabled for observability")
    
    def demo_1_basic_graph(self):
        """
        Demo 1: Basic State Graph
        This demonstrates the fundamental concept of a state graph with nodes and edges.
        """
        print("\n" + "="*60)
        print("DEMO 1: Basic State Graph")
        print("="*60)
        
        # Define the state structure
        class BasicState(TypedDict):
            messages: Annotated[List[BaseMessage], add_messages]
            current_step: str
        
        # Define nodes (functions that process the state)
        def start_node(state: BasicState) -> BasicState:
            """Initial node that sets up the conversation"""
            print("📍 Starting conversation...")
            return {
                "messages": [SystemMessage(content="You are a helpful assistant.")],
                "current_step": "started"
            }
        
        def process_node(state: BasicState) -> BasicState:
            """Process user input and generate response"""
            print("🔄 Processing user input...")
            
            # Get the last user message
            user_message = state["messages"][-1].content if state["messages"] else "Hello!"
            
            # Generate response using LLM
            response = self.llm.invoke([
                SystemMessage(content="You are a helpful assistant. Respond briefly and cheerfully."),
                HumanMessage(content=f"User said: {user_message}")
            ])
            
            # Add response to messages
            new_messages = state["messages"] + [response]
            
            return {
                "messages": new_messages,
                "current_step": "processed"
            }
        
        def end_node(state: BasicState) -> BasicState:
            """Final node that concludes the conversation"""
            print("✅ Conversation completed!")
            return {
                "messages": state["messages"] + [SystemMessage(content="Conversation ended.")],
                "current_step": "ended"
            }
        
        # Build the graph
        workflow = StateGraph(BasicState)
        
        # Add nodes
        workflow.add_node("start", start_node)
        workflow.add_node("process", process_node)
        workflow.add_node("end", end_node)
        
        # Define edges (how nodes connect)
        workflow.set_entry_point("start")
        workflow.add_edge("start", "process")
        workflow.add_edge("process", "end")
        workflow.add_edge("end", END)
        
        # Compile the graph
        app = workflow.compile()
        
        # Test the graph
        print("Testing basic state graph...")
        result = app.invoke({
            "messages": [HumanMessage(content="Hello, how are you?")],
            "current_step": ""
        })
        
        print(f"Final state: {result['current_step']}")
        print(f"Number of messages: {len(result['messages'])}")
        return result
    
    def demo_2_conditional_graph(self):
        """
        Demo 2: Conditional Graph with Decision Making
        This demonstrates how to create graphs that make decisions based on state.
        """
        print("\n" + "="*60)
        print("DEMO 2: Conditional Graph with Decision Making")
        print("="*60)
        
        class ConditionalState(TypedDict):
            messages: Annotated[List[BaseMessage], add_messages]
            user_intent: str
            confidence: float
        
        def analyze_intent(state: ConditionalState) -> ConditionalState:
            """Analyze user intent and determine next action"""
            print("🧠 Analyzing user intent...")
            
            user_message = state["messages"][-1].content if state["messages"] else ""
            
            # Simple intent classification
            if "help" in user_message.lower():
                intent = "help_request"
                confidence = 0.9
            elif "joke" in user_message.lower():
                intent = "joke_request"
                confidence = 0.8
            else:
                intent = "general_chat"
                confidence = 0.6
            
            return {
                "messages": state["messages"],
                "user_intent": intent,
                "confidence": confidence
            }
        
        def provide_help(state: ConditionalState) -> ConditionalState:
            """Provide help information"""
            print("🆘 Providing help...")
            response = self.llm.invoke([
                SystemMessage(content="You are a helpful assistant. Provide a brief, friendly help message."),
                HumanMessage(content="User needs help")
            ])
            
            return {
                "messages": state["messages"] + [response],
                "user_intent": state["user_intent"],
                "confidence": state["confidence"]
            }
        
        def tell_joke(state: ConditionalState) -> ConditionalState:
            """Tell a joke"""
            print("😄 Telling a joke...")
            response = self.llm.invoke([
                SystemMessage(content="You are a funny assistant. Tell a short, clean joke."),
                HumanMessage(content="Tell me a joke")
            ])
            
            return {
                "messages": state["messages"] + [response],
                "user_intent": state["user_intent"],
                "confidence": state["confidence"]
            }
        
        def general_chat(state: ConditionalState) -> ConditionalState:
            """Handle general conversation"""
            print("💬 General chat...")
            user_message = state["messages"][-1].content if state["messages"] else ""
            response = self.llm.invoke([
                SystemMessage(content="You are a friendly assistant. Respond naturally to the user."),
                HumanMessage(content=user_message)
            ])
            
            return {
                "messages": state["messages"] + [response],
                "user_intent": state["user_intent"],
                "confidence": state["confidence"]
            }
        
        def route_by_intent(state: ConditionalState) -> str:
            """Route to different nodes based on intent"""
            intent = state["user_intent"]
            if intent == "help_request":
                return "provide_help"
            elif intent == "joke_request":
                return "tell_joke"
            else:
                return "general_chat"
        
        # Build the conditional graph
        workflow = StateGraph(ConditionalState)
        
        # Add nodes
        workflow.add_node("analyze", analyze_intent)
        workflow.add_node("provide_help", provide_help)
        workflow.add_node("tell_joke", tell_joke)
        workflow.add_node("general_chat", general_chat)
        
        # Define edges with conditional routing
        workflow.set_entry_point("analyze")
        workflow.add_conditional_edges(
            "analyze",
            route_by_intent,
            {
                "provide_help": "provide_help",
                "tell_joke": "tell_joke",
                "general_chat": "general_chat"
            }
        )
        workflow.add_edge("provide_help", END)
        workflow.add_edge("tell_joke", END)
        workflow.add_edge("general_chat", END)
        
        # Compile the graph
        app = workflow.compile()
        
        # Test with different inputs
        test_messages = [
            "I need help with something",
            "Tell me a joke",
            "How's the weather today?"
        ]
        
        for message in test_messages:
            print(f"\nTesting with: '{message}'")
            result = app.invoke({
                "messages": [HumanMessage(content=message)],
                "user_intent": "",
                "confidence": 0.0
            })
            print(f"Intent: {result['user_intent']} (confidence: {result['confidence']:.2f})")
        
        return app
    
    def demo_3_tools_and_agents(self):
        """
        Demo 3: Tools and Agents in LangGraph
        This demonstrates how to use tools and create agent-like behavior.
        """
        print("\n" + "="*60)
        print("DEMO 3: Tools and Agents in LangGraph")
        print("="*60)
        
        # Define tools
        @tool
        def calculator(expression: str) -> str:
            """Calculate mathematical expressions safely"""
            try:
                # Only allow basic math operations for safety
                allowed_chars = set('0123456789+-*/.() ')
                if not all(c in allowed_chars for c in expression):
                    return "Error: Only basic math operations allowed"
                
                result = eval(expression)
                return f"Result: {result}"
            except Exception as e:
                return f"Error calculating: {str(e)}"
        
        @tool
        def get_current_time() -> str:
            """Get the current time"""
            from datetime import datetime
            return f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        # Initialize search tool
        search = DuckDuckGoSearchRun()
        
        class AgentState(TypedDict):
            messages: Annotated[List[BaseMessage], add_messages]
            tool_results: List[str]
            current_step: str
        
        def agent_node(state: AgentState) -> AgentState:
            """Main agent node that decides what to do"""
            print("🤖 Agent thinking...")
            
            # Get the last user message
            user_message = state["messages"][-1].content if state["messages"] else ""
            
            # Simple tool selection logic
            if any(word in user_message.lower() for word in ["calculate", "math", "compute"]):
                # Use calculator
                tool_name = "calculator"
                tool_input = user_message.replace("calculate", "").replace("math", "").strip()
                tool_result = calculator.invoke({"expression": tool_input})
            elif any(word in user_message.lower() for word in ["time", "clock"]):
                # Use time tool
                tool_name = "get_current_time"
                tool_result = get_current_time.invoke({})
            elif any(word in user_message.lower() for word in ["search", "find", "look up"]):
                # Use search tool
                tool_name = "search"
                search_query = user_message.replace("search", "").replace("find", "").replace("look up", "").strip()
                tool_result = search.invoke(search_query)
            else:
                # General conversation
                tool_name = "chat"
                response = self.llm.invoke([
                    SystemMessage(content="You are a helpful assistant. Respond naturally."),
                    HumanMessage(content=user_message)
                ])
                tool_result = response.content
            
            # Generate response based on tool result
            if tool_name != "chat":
                response = self.llm.invoke([
                    SystemMessage(content="You are a helpful assistant. Explain the tool result clearly."),
                    HumanMessage(content=f"Tool '{tool_name}' returned: {tool_result}. Explain this to the user.")
                ])
            else:
                response = self.llm.invoke([
                    SystemMessage(content="You are a helpful assistant."),
                    HumanMessage(content=user_message)
                ])
            
            return {
                "messages": state["messages"] + [response],
                "tool_results": state["tool_results"] + [str(tool_result)],
                "current_step": f"used_{tool_name}"
            }
        
        # Build the agent graph
        workflow = StateGraph(AgentState)
        workflow.add_node("agent", agent_node)
        workflow.set_entry_point("agent")
        workflow.add_edge("agent", END)
        
        app = workflow.compile()
        
        # Test the agent
        test_queries = [
            "Calculate 15 * 8 + 12",
            "What time is it?",
            "Hello, how are you?"
        ]
        
        for query in test_queries:
            print(f"\nTesting agent with: '{query}'")
            result = app.invoke({
                "messages": [HumanMessage(content=query)],
                "tool_results": [],
                "current_step": "start"
            })
            print(f"Step: {result['current_step']}")
            print(f"Tool results: {len(result['tool_results'])}")
        
        return app
    
    def demo_4_memory_and_checkpoints(self):
        """
        Demo 4: Memory and Checkpoints
        This demonstrates how to maintain state across multiple interactions.
        """
        print("\n" + "="*60)
        print("DEMO 4: Memory and Checkpoints")
        print("="*60)
        
        class MemoryState(TypedDict):
            messages: Annotated[List[BaseMessage], add_messages]
            conversation_count: int
            user_preferences: Dict[str, Any]
        
        def memory_node(state: MemoryState) -> MemoryState:
            """Node that maintains conversation memory"""
            print(f"💭 Processing conversation #{state['conversation_count']}")
            
            user_message = state["messages"][-1].content if state["messages"] else ""
            
            # Extract user preferences from conversation
            preferences = state.get("user_preferences", {})
            
            # Simple preference extraction
            if "name" in user_message.lower() and "my name is" in user_message.lower():
                # Extract name
                import re
                name_match = re.search(r"my name is (\w+)", user_message.lower())
                if name_match:
                    preferences["user_name"] = name_match.group(1)
            
            if "like" in user_message.lower() and "food" in user_message.lower():
                preferences["likes_food"] = True
            
            # Generate personalized response
            context = f"User preferences: {preferences}. "
            if "user_name" in preferences:
                context += f"User's name is {preferences['user_name']}. "
            
            response = self.llm.invoke([
                SystemMessage(content=f"You are a helpful assistant. {context}Respond naturally and remember user preferences."),
                HumanMessage(content=user_message)
            ])
            
            return {
                "messages": state["messages"] + [response],
                "conversation_count": state["conversation_count"] + 1,
                "user_preferences": preferences
            }
        
        # Build the memory graph with checkpoint
        workflow = StateGraph(MemoryState)
        workflow.add_node("memory", memory_node)
        workflow.set_entry_point("memory")
        workflow.add_edge("memory", END)
        
        # Add memory saver for persistence
        memory = MemorySaver()
        app = workflow.compile(checkpointer=memory)
        
        # Test with conversation thread
        thread_id = "demo_conversation"
        
        print("Starting conversation with memory...")
        
        # First interaction
        result1 = app.invoke(
            {"messages": [HumanMessage(content="Hi, my name is Alice")], "conversation_count": 0, "user_preferences": {}},
            config={"configurable": {"thread_id": thread_id}}
        )
        print(f"After 1st message: {result1['conversation_count']} conversations")
        print(f"Preferences: {result1['user_preferences']}")
        
        # Second interaction (should remember the name)
        result2 = app.invoke(
            {"messages": [HumanMessage(content="What's my name?")], "conversation_count": 0, "user_preferences": {}},
            config={"configurable": {"thread_id": thread_id}}
        )
        print(f"After 2nd message: {result2['conversation_count']} conversations")
        print(f"Preferences: {result2['user_preferences']}")
        
        # Third interaction
        result3 = app.invoke(
            {"messages": [HumanMessage(content="I like pizza and pasta")], "conversation_count": 0, "user_preferences": {}},
            config={"configurable": {"thread_id": thread_id}}
        )
        print(f"After 3rd message: {result3['conversation_count']} conversations")
        print(f"Preferences: {result3['user_preferences']}")
        
        return app
    
    def demo_5_parallel_processing(self):
        """
        Demo 5: Parallel Processing
        This demonstrates how to process multiple tasks in parallel.
        """
        print("\n" + "="*60)
        print("DEMO 5: Parallel Processing")
        print("="*60)
        
        class ParallelState(TypedDict):
            messages: Annotated[List[BaseMessage], add_messages]
            parallel_results: List[str]
            final_summary: str
        
        def parallel_node(state: ParallelState) -> ParallelState:
            """Process multiple tasks in parallel"""
            print("🔄 Processing tasks in parallel...")
            
            user_message = state["messages"][-1].content if state["messages"] else ""
            
            # Define parallel tasks
            tasks = [
                f"Analyze sentiment of: {user_message}",
                f"Extract key topics from: {user_message}",
                f"Summarize in one sentence: {user_message}"
            ]
            
            # Process tasks in parallel (simulated)
            results = []
            for task in tasks:
                response = self.llm.invoke([
                    SystemMessage(content="You are a helpful assistant. Complete the task briefly."),
                    HumanMessage(content=task)
                ])
                results.append(response.content)
            
            return {
                "messages": state["messages"],
                "parallel_results": results,
                "final_summary": "Parallel processing completed"
            }
        
        def summary_node(state: ParallelState) -> ParallelState:
            """Create a summary of parallel results"""
            print("📝 Creating summary...")
            
            summary_prompt = f"""
            Parallel analysis results:
            1. Sentiment: {state['parallel_results'][0]}
            2. Topics: {state['parallel_results'][1]}
            3. Summary: {state['parallel_results'][2]}
            
            Create a brief overall summary of these results.
            """
            
            response = self.llm.invoke([
                SystemMessage(content="You are a helpful assistant. Create concise summaries."),
                HumanMessage(content=summary_prompt)
            ])
            
            return {
                "messages": state["messages"] + [response],
                "parallel_results": state["parallel_results"],
                "final_summary": response.content
            }
        
        # Build the parallel processing graph
        workflow = StateGraph(ParallelState)
        workflow.add_node("parallel", parallel_node)
        workflow.add_node("summary", summary_node)
        workflow.set_entry_point("parallel")
        workflow.add_edge("parallel", "summary")
        workflow.add_edge("summary", END)
        
        app = workflow.compile()
        
        # Test parallel processing
        test_message = "I love learning about artificial intelligence and machine learning!"
        print(f"Testing parallel processing with: '{test_message}'")
        
        result = app.invoke({
            "messages": [HumanMessage(content=test_message)],
            "parallel_results": [],
            "final_summary": ""
        })
        
        print(f"Parallel results: {len(result['parallel_results'])}")
        print(f"Final summary: {result['final_summary']}")
        
        return app
    
    def run_all_demos(self):
        """Run all LangGraph demonstration examples"""
        print("🎓 Welcome to LangGraph Beginner Tutorial!")
        print("This tutorial covers 5 key LangGraph concepts:")
        print("1. Basic State Graph")
        print("2. Conditional Graph with Decision Making")
        print("3. Tools and Agents")
        print("4. Memory and Checkpoints")
        print("5. Parallel Processing")
        print("\n📊 LangSmith tracing is enabled for observability")
        
        try:
            # Run all demos
            self.demo_1_basic_graph()
            self.demo_2_conditional_graph()
            self.demo_3_tools_and_agents()
            self.demo_4_memory_and_checkpoints()
            self.demo_5_parallel_processing()
            
            print("\n" + "="*60)
            print("🎉 All LangGraph demos completed successfully!")
            print("="*60)
            print("\nKey LangGraph Concepts Learned:")
            print("• State Graphs: Building stateful applications")
            print("• Nodes: Functions that process state")
            print("• Edges: Connections between nodes")
            print("• Conditional Routing: Making decisions based on state")
            print("• Tools: External functions that agents can use")
            print("• Memory: Maintaining state across interactions")
            print("• Parallel Processing: Handling multiple tasks simultaneously")
            print("\nNext Steps:")
            print("• Explore LangSmith traces to understand execution flow")
            print("• Build your own custom graphs")
            print("• Integrate with your existing LangChain applications")
            
        except Exception as e:
            print(f"❌ Error running demos: {e}")
            print("Make sure all required packages are installed:")
            print("pip install langgraph langchain-openai langchain-community")

if __name__ == "__main__":
    # Initialize and run the tutorial
    tutorial = LangGraphTutorial()
    tutorial.run_all_demos()
