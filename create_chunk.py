import re
from langchain.text_splitter import RecursiveCharacterTextSplitter

def clean_text(text):
    text = re.sub(r'\n+', '\n', text)  # Collapse multiple newlines to one
    text = re.sub(r'\s+', ' ', text)   # Collapse multiple spaces/tabs to one
    return text.strip()  # Remove leading/trailing whitespace

def split_into_sentences(text):
    # Basic Bangla sentence splitter using '।', '!', '?'
    sentences = re.split(r'(?<=[।!?])\s+', text)
    # Filter out very short sentences (less than 10 characters)
    return [s.strip() for s in sentences if s.strip() and len(s.strip()) >= 10]

def clean_chunk(chunk):
    # Remove extra spaces (including between words and sentences)
    chunk = re.sub(r'\s+', ' ', chunk.strip())
    bengali_punctuation = r'[।""!?ৎঃ৺৻ৼ৽৲৳৴৵৶৸৹]'
    chunk = re.sub(bengali_punctuation, '', chunk)
    return chunk.strip()

def create_chunks(text, chunk_size=1000, chunk_overlap=100):
    # Use langchain text splitter for consistent chunk sizes
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,  # Approx. 500 characters
        chunk_overlap=chunk_overlap,  # 50 characters overlap
        length_function=len,
        separators=["।", "!", "?", "\n", " "]
    )
    chunks = text_splitter.split_text(text)
    # Clean each chunk to remove extra spaces and Bengali punctuation
    cleaned_chunks = [clean_chunk(chunk) for chunk in chunks if clean_chunk(chunk)]
    return cleaned_chunks

# Load the cleaned text
with open(r"F:\10mstask\env\cleaned_bangla_ocr_output.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Clean the text (remove extra lines and normalize whitespace)
text = clean_text(text)
print(f"Text cleaned, length: {len(text)} characters")

# Split into sentences
sentences = split_into_sentences(text)
print(f"Sentences created: {len(sentences)}")

# Create and clean chunks
chunks = create_chunks(text, chunk_size=1000, chunk_overlap=100)
print(f"Chunks created: {len(chunks)}")

# Print first few chunks for inspection
for i, chunk in enumerate(chunks[:5]):
    print(f"Chunk {i+1}: {chunk}")

# Save chunks to file
with open("bangla_chunks_new.txt", "w", encoding="utf-8") as f:
    for i, chunk in enumerate(chunks):
        f.write(f"[Chunk {i+1}]\n{chunk}\n\n")

print(f"Created {len(chunks)} chunks and saved to 'bangla_chunks_v2.txt'")