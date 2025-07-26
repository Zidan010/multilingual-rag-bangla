from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
import json
import faiss
import numpy as np
from fuzzywuzzy import fuzz
from sentence_transformers import SentenceTransformer, util
from groq import Groq
from collections import defaultdict, deque

# ==== Config ====
GROQ_API_KEY = "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"
MODEL_NAME = "llama3-70b-8192"
TOP_K = 3

# ==== FastAPI App ====
app = FastAPI(title="Multilingual RAG System")

# ==== Load Models, Index ====
retriever_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
sim_model = retriever_model  # same for evaluation
index = faiss.read_index("bangla_faiss_mpnet.index")
with open("bangla_faiss_metadata_mpnet.json", "r", encoding="utf-8") as f:
    chunk_data = json.load(f)

groq_client = Groq(api_key=GROQ_API_KEY)

# ==== Helper Functions ====

def search(query: str, top_k: int = TOP_K) -> List[str]:
    query_vec = retriever_model.encode([query])
    faiss.normalize_L2(query_vec)
    D, I = index.search(query_vec, top_k)
    return [chunk_data[i]["text"] for i in I[0]]

def generate_answer_groq(query: str, context_chunks: List[str], model=MODEL_NAME) -> str:
    prompt = f"""You are a helpful assistant. Answer the question using only the information from the context below. And answer only in bengali and no extra explanation.

User Question: {query}

Context:
{chr(10).join(context_chunks)}

Answer:"""
    response = groq_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()

def evaluate_case(query: str, expected: str) -> dict:
    chunks = search(query)
    generated = generate_answer_groq(query, chunks)

    fuzzy_score = fuzz.partial_ratio(generated, expected)
    sem_score = util.pytorch_cos_sim(
        sim_model.encode(generated, convert_to_tensor=True),
        sim_model.encode(expected, convert_to_tensor=True)
    ).item()
    grounded = any(expected in chunk for chunk in chunks)

    return {
        "query": query,
        "expected": expected,
        "generated": generated,
        "fuzzy_score": fuzzy_score,
        "semantic_score": round(sem_score, 3),
        "grounded": grounded
    }

# ==== Schemas ====

class QueryRequest(BaseModel):
    query: str

class EvaluationItem(BaseModel):
    query: str
    expected_answer: str

class EvaluationRequest(BaseModel):
    cases: List[EvaluationItem]


chat_memory = defaultdict(lambda: deque(maxlen=5))  # 5-turn short-term memory per user

@app.post("/ask")
def ask_question(data: QueryRequest):
    try:
        user_id = "default"  # You can extend this with real user/session IDs
        chunks = search(data.query)

        # Get previous turns (short-term memory)
        history = list(chat_memory[user_id])
        short_term = "\n".join([f"User: {q}\nBot: {a}" for q, a in history])

        # Prepare enriched prompt with memory + context
        prompt = f"""You are a helpful assistant. Use the prior conversation and the following context to answer.

Chat History:
{short_term}

User Question: {data.query}

Context:
{chr(10).join(chunks)}

Answer:"""

        response = groq_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=512,
        )
        answer = response.choices[0].message.content.strip()

        # Update memory
        chat_memory[user_id].append((data.query, answer))

        return {
            "query": data.query,
            "answer": answer,
            "context_used": chunks,
            "short_term_memory": list(chat_memory[user_id])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evaluate")
def evaluate_rag(data: EvaluationRequest):
    results = []
    for case in data.cases:
        result = evaluate_case(case.query, case.expected_answer)
        results.append(result)
    return {
        "total": len(results),
        "results": results
    }
