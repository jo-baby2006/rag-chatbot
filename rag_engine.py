import chromadb
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from groq import Groq
import os

# ── Lazy-loaded globals (initialised once per process) ────────────────────────
_embedder = None
_collection = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def _get_collection():
    global _collection
    if _collection is None:
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        _collection = chroma_client.get_or_create_collection("documents")
    return _collection


# ── PDF helpers ───────────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from a PDF file."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    """
    Split text into overlapping word-level chunks.

    FIX: Original code stepped by (chunk_size - overlap) which is correct, but
    didn't guard against chunk_size <= overlap (infinite loop).  Also capped
    chunk size so the last slice never exceeds chunk_size words.
    """
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    words = text.split()
    if not words:
        return []

    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i: i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


# ── Ingestion ─────────────────────────────────────────────────────────────────

def ingest_document(pdf_path: str, doc_name: str) -> int:
    """
    Extract text from a PDF, embed it in chunks, and store in ChromaDB.
    Returns the number of chunks stored.
    """
    raw_text = extract_text_from_pdf(pdf_path)
    if not raw_text.strip():
        raise ValueError("The PDF appears to be empty or contains no extractable text.")

    chunks = chunk_text(raw_text)
    if not chunks:
        raise ValueError("No text chunks could be created from the document.")

    embedder = _get_embedder()
    collection = _get_collection()

    embeddings = embedder.encode(chunks).tolist()
    ids = [f"{doc_name}_chunk_{i}" for i in range(len(chunks))]

    # upsert is idempotent – safe to re-ingest the same document
    collection.upsert(
        documents=chunks,
        embeddings=embeddings,
        ids=ids,
        metadatas=[{"source": doc_name}] * len(chunks),
    )
    return len(chunks)


# ── Querying ──────────────────────────────────────────────────────────────────

def query_documents(question: str, top_k: int = 3) -> tuple:
    """
    Embed the question, retrieve the top-k most similar chunks from ChromaDB,
    then send them + the question to Groq for a grounded answer.

    FIX 1: Guard against missing GROQ_API_KEY.
    FIX 2: Guard against empty collection (count() == 0).
    FIX 3: Clamp n_results to the actual collection size to avoid ChromaDB error.

    Returns (answer: str, context_chunks: list[str])
    """
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is not set. Please enter your Groq API key in the sidebar."
        )

    collection = _get_collection()

    # Guard: collection must have documents before querying
    doc_count = collection.count()
    if doc_count == 0:
        raise ValueError(
            "No documents have been ingested yet. "
            "Please upload and ingest a PDF first."
        )

    # Clamp top_k so ChromaDB doesn't raise an error when fewer docs exist
    effective_k = min(top_k, doc_count)

    embedder = _get_embedder()
    question_embedding = embedder.encode([question]).tolist()

    results = collection.query(
        query_embeddings=question_embedding,
        n_results=effective_k,
    )

    context_chunks = results["documents"][0]
    if not context_chunks:
        return "I couldn't find relevant information in the document.", []

    context = "\n\n---\n\n".join(context_chunks)

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. "
                    "Answer questions using ONLY the context provided. "
                    "If the answer is not in the context, say "
                    "'I couldn't find that in the document.'"
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            },
        ],
    )

    answer = response.choices[0].message.content
    return answer, context_chunks
