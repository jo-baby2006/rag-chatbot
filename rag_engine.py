import chromadb
import fitz
from sentence_transformers import SentenceTransformer
from groq import Groq
import os

embedder = SentenceTransformer("all-MiniLM-L6-v2")

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection("documents")

def extract_text_from_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

def ingest_document(pdf_path: str, doc_name: str):
    raw_text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(raw_text)
    embeddings = embedder.encode(chunks).tolist()
    ids = [f"{doc_name}_chunk_{i}" for i in range(len(chunks))]
    collection.upsert(
        documents=chunks,
        embeddings=embeddings,
        ids=ids,
        metadatas=[{"source": doc_name}] * len(chunks)
    )
    return len(chunks)

def query_documents(question: str, top_k: int = 3):
    question_embedding = embedder.encode([question]).tolist()
    results = collection.query(
        query_embeddings=question_embedding,
        n_results=top_k
    )
    context_chunks = results["documents"][0]
    context = "\n\n---\n\n".join(context_chunks)

    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. Answer questions using ONLY the context provided. If the answer is not in the context, say 'I couldn't find that in the document.'"
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}"
            }
        ]
    )
    answer = response.choices[0].message.content
    return answer, context_chunks
