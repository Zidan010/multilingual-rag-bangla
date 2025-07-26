import faiss
import numpy as np
import json

# Load the embeddings
with open(r"F:\10mstask\env\bangla_embeddings_mpnet.json", "r", encoding="utf-8") as f:
    chunk_data = json.load(f)

# Extract embeddings and convert to float32 NumPy array
embedding_vectors = np.array([item["embedding"] for item in chunk_data]).astype("float32")
print(f"Embeddings loaded: {len(embedding_vectors)}")

# Create FAISS index (cosine similarity)
dimension = embedding_vectors.shape[1]
index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity

# Normalize vectors for cosine similarity
faiss.normalize_L2(embedding_vectors)

# Add vectors to index
index.add(embedding_vectors)
print(f"FAISS index size: {index.ntotal}")

# Save index and metadata
faiss.write_index(index, "bangla_faiss_mpnet.index")
with open("bangla_faiss_metadata_mpnet.json", "w", encoding="utf-8") as f:
    json.dump(chunk_data, f, ensure_ascii=False, indent=2)

print(f"✅ Stored {index.ntotal} chunks into FAISS index.")

