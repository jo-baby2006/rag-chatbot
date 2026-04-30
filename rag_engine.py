import chromadb
import fitz
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import os

embedder = SentenceTransformer("all-MiniLM-L6-v2")

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection("documents")

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def ingest_document(pdf_path, doc_name):
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)
    embeddings = embedder.encode(chunks).tolist()

    ids = [f"{doc_name}_{i}" for i in range(len(chunks))]

    collection.upsert(
        documents=chunks,
        embeddings=embeddings,
        ids=ids
    )

    return len(chunks)

def query_documents(question):
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])

    question_embedding = embedder.encode([question]).tolist()

    results = collection.query(
        query_embeddings=question_embedding,
        n_results=3
    )

    context = "\n".join(results["documents"][0])

    model = genai.GenerativeModel("gemini-2.0-flash")

    prompt = f"""
Answer using only the context below:

{context}

Question: {question}
"""

    response = model.generate_content(prompt)

    return response.text, results["documents"][0]
