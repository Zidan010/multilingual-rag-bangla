from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json

# Load everything
# model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
# model = SentenceTransformer('sentence-transformers/LaBSE')
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

index = faiss.read_index("bangla_faiss_mpnet.index")
with open("bangla_faiss_metadata_mpnet.json", "r", encoding="utf-8") as f:
    chunk_data = json.load(f)

# Search function
def search(query, top_k=3):
    query_vec = model.encode([query])
    faiss.normalize_L2(query_vec)
    # faiss.IndexFlatIP(query_vec)

    D, I = index.search(query_vec, top_k)

    results = []
    for idx in I[0]:
        results.append(chunk_data[idx]["text"])
    return results



# Example search
query = "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?"
# query = "অনুপমের বয়স কত?"
# query = "মেয়ের বয়স কত?"

top_chunks = search(query)

for i, chunk in enumerate(top_chunks, 1):
    print(f"\n🔹 Top Match {i}:\n{chunk}")
    