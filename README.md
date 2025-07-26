# Multilingual RAG System for Bengali and English Queries

This repository implements a **Multilingual Retrieval-Augmented Generation (RAG)** system designed to process queries in **English** and **Bengali**, retrieve relevant information from a PDF document corpus (specifically the **HSC26 Bangla 1st Paper** book), and generate meaningful answers using a combination of **LangChain**, **FAISS**, and a large language model (**LLM**). The system includes a **FastAPI-based REST API** for interaction and an evaluation framework to assess retrieval and generation quality.

---

## 🎯 Objective

The goal is to build a multilingual RAG pipeline that:

- Accepts user queries in English and Bengali.
- Retrieves relevant document chunks from a knowledge base.
- Generates grounded answers based on retrieved content.
- Maintains short-term (chat history) and long-term (document corpus) memory.
- Provides a lightweight REST API for interaction.

---

## 📁 Project Structure

| File / Folder        | Description |
|----------------------|-------------|
| `pdf_process.py`     | Extracts text from PDF files using PyPDF2. |
| `clean.py`           | Cleans extracted text by removing unwanted characters and formatting issues. |
| `chunking.py`        | Splits cleaned text into chunks using LangChain’s `RecursiveCharacterTextSplitter`. |
| `create_embedding.py`| Generates embeddings for document chunks using Hugging Face’s `sentence-transformers`. |
| `indexing.py`        | Indexes embeddings into a FAISS vector store for retrieval. |
| `retriever.py`       | Retrieves relevant chunks using cosine similarity. |
| `app.py`             | FastAPI application for user interaction. |
| `data/`              | Contains the input PDF (`HSC26_Bangla_1st_Paper.pdf`). |
| `requirements.txt`   | Lists required Python packages. |

---

## ⚙️ Setup Guide

### ✅ Prerequisites

- Python 3.8+
- Hugging Face account (optional for public models)

### 🛠 Installation

```bash
# Clone the repository
git clone https://github.com/Zidan010/multilingual-rag-bangla.git
cd multilingual-rag-bangla

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

Ensure the PDF file (`HSC26_Bangla_1st_Paper.pdf`) is placed in the `data/` directory.

### 🚀 Run the Pipeline

```bash
python pdf_process.py
python clean.py
python chunking.py
python create_embedding.py
python indexing.py
```

### 🌐 Launch API

```bash
uvicorn app:app --reload
```

API will be available at: [http://localhost:8000](http://localhost:8000)

---

## 🧰 Tools & Libraries Used

- **PyPDF2** – PDF text extraction  
- **LangChain** – Chunking and LLM integration  
- **Hugging Face Transformers** – Embeddings and LLM  
  - `l3cube-pune/bengali-sentence-similarity-sbert`  
  - `hassanaliemon/bn_rag_llama3-8b`  
- **FAISS** – Fast similarity search for embeddings  
- **FastAPI + Uvicorn** – API backend  
- **Sentence Transformers** – Semantic vector generation  
- **Others** – `numpy`, `pandas`, `re`, `pydantic`

---

## 🔍 Sample Queries & Outputs

### 🔤 Bengali

| Query | Output |
|-------|--------|
| অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে? | শুম্ভুনাথ |
| কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে? | মামাকে |
| বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল? | ১৫ বছর |

### 🇬🇧 English

| Query | Output |
|-------|--------|
| Who is referred to as a handsome man in Anupam's words? | Shumbhunath |
| Who is mentioned as Anupam's fate deity? | Uncle (Mama) |
| What was Kalyani's actual age at the time of marriage? | 15 years |

---

## 🧪 API Documentation

**Endpoint:** `POST /rag`  
**Description:** Accepts a query and returns a generated answer along with supporting document chunks.

### 📨 Request Body
```json
{
  "query": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?"
}
```

### ✅ Response
```json
{
  "query": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?",
  "response": "শুম্ভুনাথ",
  "retrieved_chunks": [
    "অনুপমের ভাষায় শুম্ভুনাথকে সুপুরুষ বলা হয়েছে।"
  ]
}
```

### 💡 cURL Example
```bash
curl -X POST "http://localhost:8000/rag" -H "Content-Type: application/json" -d '{"query": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?"}'
```

---

## 📊 RAG Evaluation

### 📏 Metrics

- **Groundedness**: Whether answers are backed by retrieved text.
- **Relevance**: Based on cosine similarity between query and chunk embeddings (threshold: 0.7).

### 📈 Results

- **Groundedness**: 90% (based on 10 sample queries)
- **Avg. Cosine Similarity**: 0.82

| Query | Expected | Answer | Grounded? | Similarity |
|-------|----------|--------|-----------|------------|
| অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে? | শুম্ভুনাথ | শুম্ভুনাথ | ✅ | 0.85 |
| কাকে অনুপমের ভাগ্য দেবতা বলা হয়েছে? | মামাকে | মামাকে | ✅ | 0.80 |
| বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল? | ১৫ বছর | ১৫ বছর | ✅ | 0.83 |

---

## 📥 Submission Questions

### 1. **Text Extraction Method**
- **Library**: PyPDF2  
- **Why**: Lightweight and integrates well with Python pipelines  
- **Challenges**: Bengali character encoding and word merging fixed in `clean.py` with regex

### 2. **Chunking Strategy**
- **Method**: LangChain’s `RecursiveCharacterTextSplitter`  
- **Config**: Chunk size: 500, Overlap: 50  
- **Reason**: Preserves context across chunk boundaries, crucial for Bengali

### 3. **Embedding Model**
- **Model**: `l3cube-pune/bengali-sentence-similarity-sbert`  
- **Why**: Fine-tuned for Bengali; better than general multilingual models  
- **How**: Uses Sentence-BERT to produce 768-dimensional semantic vectors

### 4. **Similarity Method & Storage**
- **Comparison**: Cosine similarity  
- **Why**: Angular distance is robust for embeddings  
- **Storage**: FAISS – fast, scalable, optimized for high-dimensional vector search

### 5. **Meaningful Comparison**
- **How**: Embeddings from sentence-similarity model + top-k retrieval  
- **Handling Vagueness**: System may return incorrect info for vague queries; improvements could include query rewriting or clarification prompts

### 6. **Relevance & Improvements**
- **Current**: Highly relevant (90% groundedness, 0.82 similarity)  
- **Suggestions**:
  - Smaller chunk sizes
  - Better embedding models (e.g., `paraphrase-multilingual-mpnet-base-v2`)
  - Larger corpus with diverse topics
  - Query expansion techniques

---

## 🧠 Memory Management

- **Short-Term**: Chat history maintained in `app.py`  
- **Long-Term**: Full document vector store (FAISS) retained across sessions
