from sentence_transformers import SentenceTransformer
import json
import re

# Load model
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

# Load chunks
chunks = []
with open(r"F:\10mstask\env\bangla_chunks_new.txt", "r", encoding="utf-8") as f:
    content = f.read().strip()
    # Split by '[Chunk' and filter out empty entries
    raw_chunks = re.split(r'\[Chunk \d+\]', content)[1:]  # Skip first empty split
    chunks = [chunk.strip() for chunk in raw_chunks if chunk.strip() and len(chunk.strip()) >= 1]

print(f"Chunks loaded: {len(chunks)}")

# Create embeddings
print("🔄 Generating embeddings...")
embeddings = model.encode(chunks, batch_size=3,show_progress_bar=True)
print(f"Embeddings generated: {len(embeddings)}")

# Save as JSON
chunk_data = [{"chunk_id": i, "text": text, "embedding": emb.tolist()} for i, (text, emb) in enumerate(zip(chunks, embeddings))]

with open("bangla_embeddings_mpnet.json", "w", encoding="utf-8") as f:
    json.dump(chunk_data, f, ensure_ascii=False, indent=2)

print(f"Embeddings saved to 'bangla_embeddings_mpnet.json' with {len(chunk_data)} chunks")