# LangChain Beginner Tutorial

A hands-on application demonstrating key LangChain concepts for beginners.

## 🎯 What You'll Learn

This tutorial covers 5 essential LangChain concepts:

1. **Basic LLM Interaction** - Direct communication with language models
2. **Prompt Templates** - Reusable, parameterized prompts
3. **Conversation Memory** - Maintaining context across interactions
4. **Tools and Agents** - AI that can use external tools to solve problems
5. **RAG (Retrieval-Augmented Generation)** - Combining LLMs with external knowledge

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

3. **Run the tutorial:**
   ```bash
   python main.py
   ```

## 📚 Key Concepts Explained

### LLM (Large Language Model)
The foundation of LangChain - AI models that understand and generate text.

### Prompt Templates
Reusable prompt structures with variables, making your AI interactions consistent and flexible.

### Memory
Allows your AI to remember previous parts of the conversation, enabling more natural interactions.

### Agents & Tools
AI that can decide which tools to use to solve complex problems, like a smart assistant that knows when to use a calculator or search the web.

### RAG (Retrieval-Augmented Generation)
A technique that enhances LLM responses by retrieving relevant information from external documents, enabling AI to access up-to-date and domain-specific knowledge.

## 🛠️ Requirements

- Python 3.8+
- OpenAI API key
- Internet connection
- ~100MB disk space (for ChromaDB vector store)

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

## 📖 Next Steps

After running this tutorial, try:
- Modifying the prompt templates
- Adding new custom tools
- Experimenting with different memory types
- Building your own agent workflows