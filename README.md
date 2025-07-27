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

# 📘 API Endpoints Documentation

---

## 1. 🗨️ Ask Question

**Endpoint:** `POST /ask`  
**Description:** Accepts a query and returns a generated answer in **Bengali**, using context from retrieved chunks and short-term chat memory.

### 📥 Request Body
```json
{
  "query": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?"
}
```

### 📤 Response
```json
{
  "query": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?",
  "answer": "শস্তুনাথবাবুকে",
  "short_term_memory": [
    ["অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?", "শস্তুনাথবাবুকে"]
  ]
}
```

### 🧪 cURL Example
```bash
curl -X POST "http://localhost:8000/ask" \
-H "Content-Type: application/json" \
-d '{"query": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?"}'
```

---

## 2. 📊 Evaluate RAG

**Endpoint:** `POST /evaluate`  
**Description:** Evaluates the RAG system by comparing generated answers against expected answers. Returns fuzzy score, semantic similarity, and groundedness.

### 📥 Request Body
```json
{
  "cases": [
    {
      "query": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?",
      "expected_answer": "শস্তুনাথবাবুকে"
    }
  ]
}
```

### 📤 Response
```json
{
  "total": 1,
  "results": [
    {
      "query": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?",
      "expected": "শস্তুনাথবাবুকে",
      "generated": "শস্তুনাথবাবুকে",
      "fuzzy_score": 100,
      "semantic_score": 0.999,
      "grounded": true
    }
  ]
}
```

### 🧪 cURL Example
```bash
curl -X POST "http://localhost:8000/evaluate" \
-H "Content-Type: application/json" \
-d '{"cases": [{"query": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?", "expected_answer": "শস্তুনাথবাবুকে"}]}'
```

---

## 📝 Notes

- 🔍 The API uses the `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` model for **retrieval** and **semantic similarity evaluation**.
- 🧠 Answers are generated using the `llama3-70b-8192` model via **Groq API**, with:
  - `temperature`: `0.4`
- 💬 The system maintains **short-term memory** of up to **5 previous query-answer pairs per user** for context-aware responses.
- 📐 `/evaluate` returns the following metrics:
  - **Fuzzy Score**: Partial text match ratio (Levenshtein-based).
  - **Semantic Score**: Cosine similarity between embeddings of generated and expected answers.
  - **Grounded**: Indicates whether the expected answer was found in the retrieved context.
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
