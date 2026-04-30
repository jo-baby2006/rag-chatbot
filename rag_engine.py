import os
import chromadb
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from groq import Groq

# ---------------------------------------------------------------------------
# Globals – initialised once per process
# ---------------------------------------------------------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection("documents")


# ---------------------------------------------------------------------------
# PDF helpers
# ---------------------------------------------------------------------------
def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from a PDF file."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Split *text* into overlapping word-level chunks.

    BUG FIX: the original used `range(0, len(words), chunk_size - overlap)`
    which stepped by 450 words but still sliced 500 words per chunk, so every
    chunk contained the same 50-word tail as the next chunk's head — correct
    semantics — BUT the slice `words[i : i + chunk_size]` with step 450 gives
    the right overlap.  The real bug was that `chunk_size - overlap` was used
    as the *step*, meaning the step equalled 450, while the window was 500,
    giving a 50-word overlap.  That is actually correct mathematically, but
    only if the step < chunk_size.  We keep the formula and add a safety guard
    so that overlap is never >= chunk_size.
    """
    if not text.strip():
        return []

    overlap = min(overlap, chunk_size - 1)  # guard against bad inputs
    words = text.split()
    if not words:
        return []

    step = chunk_size - overlap          # e.g. 500 - 50 = 450
    chunks = []
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------
def ingest_document(pdf_path: str, doc_name: str) -> int:
    """
    Ingest a PDF into ChromaDB.  Returns the number of chunks stored.
    Raises ValueError if no text could be extracted.
    """
    raw_text = extract_text_from_pdf(pdf_path)
    if not raw_text.strip():
        raise ValueError(
            f"No extractable text found in '{doc_name}'. "
            "The PDF may be scanned/image-based."
        )

    chunks = chunk_text(raw_text)
    if not chunks:
        raise ValueError(f"Text extracted from '{doc_name}' produced no chunks.")

    embeddings = embedder.encode(chunks).tolist()
    ids = [f"{doc_name}_chunk_{i}" for i in range(len(chunks))]

    collection.upsert(
        documents=chunks,
        embeddings=embeddings,
        ids=ids,
        metadatas=[{"source": doc_name}] * len(chunks),
    )
    return len(chunks)


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------
def query_documents(question: str, top_k: int = 3) -> tuple[str, list[str]]:
    """
    Retrieve relevant chunks for *question* and ask Groq to answer.

    BUG FIX: ChromaDB raises an error when n_results > number of stored
    documents.  We now clamp top_k to the actual collection count.
    """
    # Guard: clamp top_k to available document count
    count = collection.count()
    if count == 0:
        return (
            "No documents have been ingested yet. Please upload a PDF first.",
            [],
        )
    top_k = min(top_k, count)

    question_embedding = embedder.encode([question]).tolist()
    results = collection.query(
        query_embeddings=question_embedding,
        n_results=top_k,
    )

    context_chunks: list[str] = results["documents"][0]
    context = "\n\n---\n\n".join(context_chunks)

    # BUG FIX: read the key at call-time (not at import time) so that
    # the Streamlit app can set os.environ["GROQ_API_KEY"] before first query.
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        return "GROQ_API_KEY is not set. Please enter your API key.", context_chunks

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Answer questions using ONLY the "
                    "context provided. If the answer is not in the context, say "
                    "'I couldn't find that in the document.'"
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            },
        ],
    )

    answer: str = response.choices[0].message.content
    return answer, context_chunks


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
def list_ingested_sources() -> list[str]:
    """Return a deduplicated list of source document names in the collection."""
    if collection.count() == 0:
        return []
    all_meta = collection.get(include=["metadatas"])["metadatas"]
    sources = sorted({m["source"] for m in all_meta if m})
    return sources
