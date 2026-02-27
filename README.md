# 📘 RAG Prototype (Retrieval-Augmented Generation)

## 🚀 What is RAG?

**Retrieval-Augmented Generation (RAG)** is a technique that combines:

- 🔎 **Information Retrieval** (searching relevant documents)
- 🧠 **Text Generation** (using a language model to generate answers)

Instead of generating answers purely from memory, a RAG system:
1. Retrieves relevant information from documents.
2. Uses that retrieved context to generate an informed response.

This improves factual accuracy and makes responses document-aware.

---

## 🧠 Project Overview

This project is a **simple, fully local RAG prototype** built using Python and open-source libraries.

It:

- Loads text documents
- Converts them into embeddings
- Stores them in a FAISS vector database
- Retrieves relevant documents for a query
- Generates answers using a local language model

✅ Runs fully offline  
✅ No paid APIs  
✅ No external services  
✅ Beginner-friendly implementation  

---

- `rag.py` → Main RAG pipeline
- `data/` → Folder containing knowledge documents

You can add more `.txt` files inside the `data` folder.

---

## ⚙️ Technologies Used

- Python 3.10
- PyTorch (CPU)
- Sentence-Transformers
- Transformers
- FAISS (Vector Search)

---

## 🔄 How It Works

1. 📄 Load documents from `data/`
2. 🔢 Convert documents into embeddings using `all-MiniLM-L6-v2`
3. 📦 Store embeddings in FAISS index
4. ❓ Accept user question
5. 🔍 Retrieve most relevant documents
6. 🧠 Generate answer using `distilgpt2`





