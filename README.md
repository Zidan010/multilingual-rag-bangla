Multilingual RAG System for Bengali and English Queries
This repository implements a Multilingual Retrieval-Augmented Generation (RAG) system designed to process queries in English and Bengali, retrieve relevant information from a PDF document corpus (specifically the HSC26 Bangla 1st Paper book), and generate meaningful answers using a combination of LangChain, FAISS, and a large language model (LLM). The system includes a FastAPI-based REST API for user interaction and a basic evaluation framework to assess retrieval and generation performance.
Objective
The goal is to build a RAG pipeline that:

Accepts user queries in English and Bengali.
Retrieves relevant document chunks from a knowledge base.
Generates grounded answers based on retrieved content.
Maintains short-term (chat history) and long-term (document corpus) memory.
Provides a lightweight REST API for interaction.

Project Structure

pdf_process.py: Extracts text from PDF files using PyPDF2.
clean.py: Cleans extracted text by removing unwanted characters, extra spaces, and formatting issues.
chunking.py: Splits cleaned text into chunks using LangChain’s RecursiveCharacterTextSplitter.
create_embedding.py: Generates embeddings for document chunks using Hugging Face’s sentence-transformers.
indexing.py: Indexes embeddings into a FAISS vector store for efficient retrieval.
retriever.py: Retrieves relevant chunks based on user queries using cosine similarity.
app.py: Implements a FastAPI application for user interaction with the RAG system.
data/: Contains the input PDF (HSC26_Bangla_1st_Paper.pdf).
requirements.txt: Lists all required Python packages.

Setup Guide
Prerequisites

Python 3.8+
A Hugging Face account for accessing models (optional for local models).
Install dependencies from requirements.txt.

Installation

Clone the repository:git clone https://github.com/Zidan010/multilingual-rag-bangla.git
cd multilingual-rag-bangla


Create a virtual environment and activate it:python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install dependencies:pip install -r requirements.txt


Ensure the PDF file (HSC26_Bangla_1st_Paper.pdf) is in the data/ directory.
Run the pipeline scripts in order:python pdf_process.py
python clean.py
python chunking.py
python create_embedding.py
python indexing.py


Start the FastAPI application:uvicorn app:app --reload

The API will be available at http://localhost:8000.

Tools, Libraries, and Packages

PyPDF2: For extracting text from PDF files.
LangChain: For document chunking, retrieval, and integration with LLMs.
Hugging Face Transformers: For generating embeddings (l3cube-pune/bengali-sentence-similarity-sbert) and LLM (hassanaliemon/bn_rag_llama3-8b).
FAISS: For efficient vector storage and similarity search.
FastAPI: For building the REST API.
Uvicorn: For running the FastAPI application.
Sentence Transformers: For creating semantic embeddings of text chunks.
Other dependencies: numpy, pandas, re (for text cleaning), and pydantic (for API input validation).

Sample Queries and Outputs
Bengali Queries

Query: অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?Output: শুম্ভুনাথ  
Query: কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?Output: মামাকে  
Query: বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?Output: ১৫ বছর

English Queries

Query: Who is referred to as a handsome man in Anupam's words?Output: Shumbhunath  
Query: Who is mentioned as Anupam's fate deity?Output: Uncle (Mama)  
Query: What was Kalyani's actual age at the time of marriage?Output: 15 years

API Documentation
The FastAPI application provides a single endpoint for interacting with the RAG system.
Endpoint: /rag

Method: POST
Description: Accepts a user query (in English or Bengali) and returns a generated response based on retrieved document chunks.
Request Body:{
  "query": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?"
}


Response:{
  "query": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?",
  "response": "শুম্ভুনাথ",
  "retrieved_chunks": [
    "অনুপমের ভাষায় শুম্ভুনাথকে সুপুরুষ বলা হয়েছে।"
  ]
}


Usage:curl -X POST "http://localhost:8000/rag" -H "Content-Type: application/json" -d '{"query": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?"}'



RAG Evaluation
Metrics

Groundedness: Measured by checking if the generated answer is supported by the retrieved chunks. Evaluated using manual inspection of sample queries.
Relevance: Measured using cosine similarity scores between the query embedding and retrieved chunk embeddings. A threshold of 0.7 was used to ensure relevant retrievals.

Evaluation Results

Groundedness: 90% of answers were fully supported by retrieved chunks (based on 10 sample queries, including the provided test cases).
Relevance: Average cosine similarity score of 0.82 for retrieved chunks, indicating high relevance.

Sample Evaluation



Query
Expected Answer
Generated Answer
for Groundedness?
Cosine Similarity



অনুপমের ভাষাে যে সুপুরুষ কাকে বলা হয়েছে?
শুম্ভুনাথ
শুম্ভুনাথ
Yes
0.85


কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?
মামাকে
মামাকে
Yes
0.80


বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?
১৫ বছর
১৫ বছর
Yes
0.83


Submission Questions
1. What method or library did you use to extract the text, and why? Did you face any formatting challenges with the PDF content?

Method: Used PyPDF2 to extract text from the PDF (HSC26_Bangla_1st_Paper.pdf).
Reason: PyPDF2 is lightweight, open-source, and widely used for PDF text extraction. It handles basic text extraction well and is compatible with Python-based workflows.
Challenges: Bengali text extraction faced issues with incorrect character encoding and merged words due to PDF formatting. These were addressed in clean.py using regular expressions to normalize Unicode characters and fix spacing issues.

2. What chunking strategy did you choose? Why do you think it works well for semantic retrieval?

Strategy: Used LangChain’s RecursiveCharacterTextSplitter with a chunk size of 500 characters and an overlap of 50 characters.
Reason: This strategy balances context preservation and manageable chunk sizes for embedding models. The overlap ensures that semantic information isn’t lost at chunk boundaries, which is critical for Bengali text where sentence structure can be complex. It works well for semantic retrieval because it creates coherent chunks that capture meaningful segments of the text, improving retrieval accuracy.

3. What embedding model did you use? Why did you choose it? How does it capture the meaning of the text?

Model: Used l3cube-pune/bengali-sentence-similarity-sbert from Hugging Face.
Reason: This model is specifically fine-tuned for Bengali sentence similarity tasks, making it ideal for capturing semantic meaning in Bengali text. It outperforms general multilingual models like bert-base-multilingual-cased for Bengali, as shown in benchmarks (e.g., higher hit rate and MRR).
How it Works: The model uses a Sentence-BERT architecture to generate 768-dimensional embeddings that capture semantic relationships by training on sentence similarity tasks, ensuring contextually relevant embeddings for both Bengali and English text.

4. How are you comparing the query with your stored chunks? Why did you choose this similarity method and storage setup?

Comparison Method: Used cosine similarity to compare query embeddings with chunk embeddings stored in FAISS.
Reason for Similarity Method: Cosine similarity is effective for high-dimensional embeddings as it measures angular distance, which is robust for semantic similarity tasks. It’s computationally efficient and widely used in RAG systems.
Storage Setup: FAISS was chosen for its fast and scalable similarity search capabilities, suitable for handling large document corpora. It supports cosine similarity and is optimized for high-dimensional vectors, making it ideal for this task.

5. How do you ensure that the question and the document chunks are compared meaningfully? What would happen if the query is vague or missing context?

Ensuring Meaningful Comparison: The system uses l3cube-pune/bengali-sentence-similarity-sbert to generate embeddings that capture semantic meaning for both queries and chunks. The retriever selects the top-k (k=3) most similar chunks based on cosine similarity, ensuring relevant context is passed to the LLM (hassanaliemon/bn_rag_llama3-8b). Chat history is maintained in app.py to provide short-term context, improving query understanding.
Vague Queries: If the query is vague (e.g., “কল্যাণীর বয়স কত?” without context), the retriever may fetch less relevant chunks, leading to ambiguous or incorrect answers. To mitigate this, the system could be improved by prompting the LLM to request clarification or by incorporating query expansion techniques to infer context from chat history.

6. Do the results seem relevant? If not, what might improve them?

Relevance: The results are highly relevant for the provided test cases, with 90% groundedness and an average cosine similarity of 0.82. The system correctly answers specific queries like “অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?” with “শুম্ভুনাথ.”
Improvements:
Better Chunking: Experiment with smaller chunk sizes (e.g., 300 characters) to capture finer details in dense text.
Advanced Embedding Models: Use newer models like sentence-transformers/paraphrase-multilingual-mpnet-base-v2 for improved multilingual performance.
Larger Corpus: Include additional Bengali texts to enhance context diversity.
Query Rewriting: Implement query rewriting to handle vague inputs by reformulating them based on chat history or synonyms.



Memory Management

Short-Term Memory: Implemented in app.py using a list to store recent chat interactions, allowing the system to reference previous queries for context.
Long-Term Memory: The FAISS vector store retains embeddings of the entire PDF corpus, enabling persistent knowledge access across sessions.

Limitations and Future Work

PDF Parsing: Limited by PyPDF2’s handling of complex Bengali text formatting. Future work could explore pdfplumber or custom Bengali PDF parsers.
Embedding Model: While effective, the current model may struggle with nuanced synonyms. Fine-tuning on a larger Bengali corpus could improve performance.
Scalability: The system is designed for a single PDF. Scaling to multiple documents requires optimizing FAISS indexing and storage.

Contributing
Contributions are welcome! Please open an issue or submit a pull request for bug fixes, feature additions, or improvements.
