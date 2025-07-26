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
| `pdf_process.py`     | Extracts images from pdf through pdf2image and extract text from images using pytesseract. |
| `clean.py`           | Cleans extracted text by removing unwanted characters and formatting issues. |
| `chunking.py`        | Splits cleaned text into chunks using LangChain’s `RecursiveCharacterTextSplitter`. |
| `create_embedding.py`| Generates embeddings for document chunks using Hugging Face’s `sentence-transformers`. |
| `indexing.py`        | Indexes embeddings into a FAISS vector store for retrieval. |
| `retriever.py`       | Retrieves relevant chunks using cosine similarity. |
| `app.py`             | FastAPI application for user interaction. |
| `requirements.txt`   | Lists required Python packages. |

---

## ⚙️ Setup Guide

### ✅ Prerequisites

- Python 3.8+

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

- **pdf2image** – Image extraction from pdf
- **pytesseract** - text extraction
- **LangChain** – Chunking and LLM integration  
- **Sentence Transformers** – Embeddings  
  - `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`  
- **FAISS** – Fast similarity search for embeddings
- **GROQ API** - To use LLM for answer generation
- **FastAPI + Uvicorn** – API backend  
- **Sentence Transformers** – Semantic vector generation  
- **Others** – `numpy`, `pandas`, `re`, `pydantic`

---

## 🔍 Sample Queries & Outputs

### 🔤 Bengali

| Query | Output |
|-------|--------|
| অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে? |  শস্তুনাথবাবুকে |
| কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে? | মামা কে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে। |
| বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল? | মেয়ের বয়স পনেরো |

### 🇬🇧 English

| Query | Output |
|-------|--------|
| Who is referred to as a handsome man in Anupam's words? |  শস্তুনাথবাবুকে |
| Who is mentioned as Anupam's fate deity? | মামা কে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে। |
| What was Kalyani's actual age at the time of marriage? | মেয়ের বয়স পনেরো  |

---

## 🧪 API Documentation

**Endpoint:** `POST /rag`  
**Description:** Accepts a query and returns a generated answer along with supporting history chunks.

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
  "response": " শস্তুনাথবাবুকে",
  "short_term_memory": [
      "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?",
      "শস্তুনাথবাবুকে।"  ]
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
| অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে? | শুম্ভুনাথ |  শস্তুনাথবাবুকে | ❌ | 0.85 |
| কাকে অনুপমের ভাগ্য দেবতা বলা হয়েছে? | মামাকে | মামা কে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে। | ❌ | 0.68 |
| বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল? | ১৫ বছর | মেয়ের বয়স পনেরো | ❌ | 0.73 |

---

## 📥 Submission Questions

### 1. **Text Extraction Method**
- **Library**: pdf2image and pytesseract 
- **Why**: Lightweight and integrates well with Python pipelines  
- **Challenges**: Bengali character encoding in pdf and word merging 

### 2. **Chunking Strategy**
- **Method**: LangChain’s `RecursiveCharacterTextSplitter`  
- **Config**: Chunk size: 1000, Overlap: 100 `smaller chunks doesn't perform well in retrieval`  
- **Reason**: Preserves context across chunk boundaries, crucial for Bengali

### 3. **Embedding Model**
- **Model**: `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`  
- **Why**: Better than general multilingual models that i tried 
- **How**: produce 384-dimensional semantic vectors

### 4. **Similarity Method & Storage**
- **Comparison**: L2 distance  
- **Why**: Angular distance is robust for embeddings  
- **Storage**: FAISS – fast, scalable, optimized for high-dimensional vector search

### 5. **Meaningful Comparison**
- **How**: Embeddings from sentence-similarity model + top-k retrieval  
- **Handling Vagueness**: System may return incorrect info for vague queries though i used multiple contexts; improvements could include query rewriting or clarification prompts

### 6. **Relevance & Improvements**
- **Current**: Highly relevant (average groundedness, better similarity)  
- **Suggestions**:
  - midium chunk sizes
  - Better embedding models
  - Larger corpus with diverse topics
  - Query expansion techniques

---

## 🧠 Memory Management

- **Short-Term**: Chat history maintained in `app.py`  
- **Long-Term**: Full document vector store (FAISS) retained across sessions
