# 📄 PDFChat: AI-Powered Document Assistant

A Python-based RAG (Retrieval-Augmented Generation) tool that allows users to chat with their PDF documents in natural language.

## 🎥 Demo

*Check out the demo!*


## 🚀 Features

RAG Architecture: Retains context from large documents using Retrieval-Augmented Generation.
Local Embeddings: Uses SentenceTransformer (HuggingFace) for free, local vector embedding.
Vector Search: Implements FAISS for efficient similarity search and retrieval.
Smart Generation: Connects to OpenAI GPT-3.5 to generate natural language answers based on retrieved context.

## 🛠️ Tech Stack

Frontend: Streamlit
LLM Framework: LangChain
Vector Store: FAISS (Facebook AI Similarity Search)
Embeddings: HuggingFace `all-MiniLM-L6-v2`
Model: OpenAI GPT-3.5 Turbo