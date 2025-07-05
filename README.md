# LangChain & LangGraph Tutorials

A comprehensive collection of hands-on applications demonstrating key LangChain and LangGraph concepts for beginners.

## 🎯 What You'll Learn

### LangChain Tutorial (`main.py`)
This tutorial covers 5 essential LangChain concepts:

1. **Basic LLM Interaction** - Direct communication with language models
2. **Prompt Templates** - Reusable, parameterized prompts
3. **Conversation Memory** - Maintaining context across interactions
4. **Tools and Agents** - AI that can use external tools to solve problems
5. **RAG (Retrieval-Augmented Generation)** - Combining LLMs with external knowledge

### LangGraph Tutorial (`langraph_tutorial.py`)
This tutorial covers 5 key LangGraph concepts for building stateful AI applications:

1. **Basic State Graph** - Building stateful applications with nodes and edges
2. **Conditional Graph with Decision Making** - Creating graphs that make decisions based on state
3. **Tools and Agents** - Using tools and creating agent-like behavior in LangGraph
4. **Memory and Checkpoints** - Maintaining state across multiple interactions
5. **Parallel Processing** - Processing multiple tasks in parallel

## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   # Windows
   python -m pip install -r requirements.txt
   # Or try:
   py -m pip install -r requirements.txt
   
   # Linux/Mac
   pip install -r requirements.txt
   ```

2. **Set up your API key:**
   ```bash
   copy .env.example .env
   # Edit .env and add your OpenAI API key
   ```

3. **Run the tutorials:**
   ```bash
   # Run LangChain tutorial
   python main.py
   
   # Run LangGraph tutorial
   python langraph_tutorial.py
   ```

## 📚 Key Concepts Explained

### LangChain Concepts

#### LLM (Large Language Model)
The foundation of LangChain - AI models that understand and generate text.

#### Prompt Templates
Reusable prompt structures with variables, making your AI interactions consistent and flexible.

#### Memory
Allows your AI to remember previous parts of the conversation, enabling more natural interactions.

#### Agents & Tools
AI that can decide which tools to use to solve complex problems, like a smart assistant that knows when to use a calculator or search the web.

#### RAG (Retrieval-Augmented Generation)
A technique that enhances LLM responses by retrieving relevant information from external documents, enabling AI to access up-to-date and domain-specific knowledge.

### LangGraph Concepts

#### State Graphs
The core of LangGraph - building stateful applications where data flows through a series of nodes (functions) connected by edges.

#### Nodes
Functions that process state and can perform actions like calling LLMs, using tools, or making decisions.

#### Edges
Connections between nodes that define how data flows through your application.

#### Conditional Routing
Making decisions based on state to route to different nodes, enabling complex workflows.

#### Tools in LangGraph
External functions that agents can use, integrated seamlessly into the graph workflow.

#### Memory and Checkpoints
Maintaining state across multiple interactions, allowing for persistent conversations and user preferences.

#### Parallel Processing
Handling multiple tasks simultaneously within the graph structure.

## 🛠️ Requirements

- Python 3.8+
- OpenAI API key
- Internet connection
- ~100MB disk space (for ChromaDB vector store)

## 📊 LangSmith Tracing

Both tutorials include LangSmith tracing for observability:
- All LLM calls are traced to your LangSmith project
- View execution flows, token usage, and performance metrics
- Configure tracing in your `.env` file:
  ```
  LANGCHAIN_TRACING_V2=true
  LANGCHAIN_API_KEY=your-langsmith-key
  LANGCHAIN_PROJECT=your-project-name
  ```

## 📚 RAG Setup

The RAG demo uses:
- **ChromaDB** - Local vector database for storing document embeddings
- **sample_docs.txt** - Sample documents about LangChain concepts
- **OpenAI Embeddings** - Converts text to vector representations

**Windows Installation:**
```bash
python -m pip install chromadb langchain-text-splitters
# Or try:
py -m pip install chromadb langchain-text-splitters
```

The app will skip RAG demo if ChromaDB isn't installed and show installation instructions.

## 🔧 LangGraph Dependencies

The LangGraph tutorial requires additional packages:
- `langgraph` - The main LangGraph framework
- `langchain-core` - Core LangChain components
- `langsmith` - For tracing and observability
- `duckduckgo-search` - For web search functionality

All dependencies are included in `requirements.txt`.

## 📖 Next Steps

After running these tutorials, try:

### LangChain
- Modifying the prompt templates
- Adding new custom tools
- Experimenting with different memory types
- Building your own agent workflows

### LangGraph
- Building custom state graphs
- Creating multi-agent systems
- Implementing complex workflows with conditional routing
- Integrating with your existing LangChain applications
- Exploring LangSmith traces to understand execution flow

## 🎓 Learning Path

1. Start with `main.py` to understand basic LangChain concepts
2. Move to `langraph_tutorial.py` to learn stateful application building
3. Combine both approaches in your own projects
4. Use LangSmith to monitor and debug your applications