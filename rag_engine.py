import os
import json
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq

embedder = SentenceTransformer("all-MiniLM-L6-v2")
EMBEDDING_DIM = 384

DB_DIR = Path("./faiss_db")
DB_DIR.mkdir(exist_ok=True)
INDEX_PATH = DB_DIR / "index.faiss"
META_PATH  = DB_DIR / "metadata.json"


def _load_store():
    if INDEX_PATH.exists() and META_PATH.exists():
        index = faiss.read_index(str(INDEX_PATH))
        with open(META_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    else:
        index = faiss.IndexFlatL2(EMBEDDING_DIM)
        metadata = []
    return index, metadata


def _save_store(index, metadata):
    faiss.write_index(index, str(INDEX_PATH))
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False)


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


def chunk_text(text, chunk_size=500, overlap=50):
    if not text.strip():
        return []
    overlap = min(overlap, chunk_size - 1)
    words = text.split()
    if not words:
        return []
    step = chunk_size - overlap
    chunks = []
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks


def ingest_document(pdf_path, doc_name):
    raw_text = extract_text_from_pdf(pdf_path)
    if not raw_text.strip():
        raise ValueError(f"No extractable text found in '{doc_name}'. The PDF may be scanned/image-based.")

    chunks = chunk_text(raw_text)
    if not chunks:
        raise ValueError(f"Text from '{doc_name}' produced no chunks.")

    embeddings = embedder.encode(chunks, show_progress_bar=False).astype("float32")
    index, metadata = _load_store()

    keep = [m for m in metadata if m["source"] != doc_name]
    if len(keep) < len(metadata):
        new_index = faiss.IndexFlatL2(EMBEDDING_DIM)
        if keep:
            kept_vecs = embedder.encode([m["text"] for m in keep], show_progress_bar=False).astype("float32")
            new_index.add(kept_vecs)
        index = new_index
        metadata = keep

    index.add(embeddings)
    for chunk in chunks:
        metadata.append({"source": doc_name, "text": chunk})

    _save_store(index, metadata)
    return len(chunks)


def query_documents(question, top_k=3):
    index, metadata = _load_store()

    if index.ntotal == 0:
        return "No documents have been ingested yet. Please upload a PDF first.", []

    top_k = min(top_k, index.ntotal)
    q_vec = embedder.encode([question], show_progress_bar=False).astype("float32")
    _, indices = index.search(q_vec, top_k)

    context_chunks = [metadata[i]["text"] for i in indices[0] if i < len(metadata)]
    context = "\n\n---\n\n".join(context_chunks)

    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        return "GROQ_API_KEY is not set. Please enter your API key.", context_chunks

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Answer questions using ONLY the context provided. If the answer is not in the context, say 'I couldn't find that in the document.'"},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )
    return response.choices[0].message.content, context_chunks


def list_ingested_sources():
    _, metadata = _load_store()
    return sorted({m["source"] for m in metadata})
