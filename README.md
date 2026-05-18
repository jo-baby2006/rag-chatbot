#  RAG Chatbot

A **Retrieval-Augmented Generation (RAG)** chatbot that lets you upload PDF documents and ask questions about them. It retrieves relevant context from your documents and generates accurate, grounded answers using the **Groq LLM API** — all through a clean **Streamlit** web interface.

---

##  Features

-  **PDF Ingestion** — Upload and parse PDF documents automatically
-  **Semantic Search** — Chunks and embeds documents using `sentence-transformers`
-  **Vector Storage** — Persists embeddings locally with **ChromaDB**
-  **LLM-Powered Answers** — Queries **LLaMA 3 (8B)** via the Groq API for fast, accurate responses
-  **Stays on Topic** — Only answers from document context; won't hallucinate outside it
-  **Simple UI** — Built with Streamlit, no frontend experience needed

---

##  How It Works

```
User uploads PDF
      ↓
Text is extracted and split into chunks
      ↓
Chunks are embedded (all-MiniLM-L6-v2) and stored in ChromaDB
      ↓
User asks a question
      ↓
Top-K relevant chunks are retrieved via semantic search
      ↓
Context + question are sent to Groq (LLaMA 3)
      ↓
Answer is displayed in the chat UI
```

---

##  Project Structure

```
rag-chatbot/
├── app.py              # Streamlit frontend — UI and API key input
├── rag_engine.py       # Core RAG logic — ingestion, chunking, embedding, querying
├── requirements.txt    # Python dependencies
└── .devcontainer/      # Dev container configuration
```

---

##  Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/jo-baby2006/rag-chatbot.git
cd rag-chatbot
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Get a Groq API Key

Sign up for a free key at [console.groq.com](https://console.groq.com).

### 4. Run the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

### 5. Use the Chatbot

1. Enter your **Groq API key** in the input field
2. Upload a **PDF document**
3. Type your **question** and get an answer grounded in the document

---

## 🛠️ Tech Stack

| Component | Library / Service |
|---|---|
| Frontend | [Streamlit](https://streamlit.io) |
| PDF Parsing | [PyMuPDF (fitz)](https://pymupdf.readthedocs.io) |
| Embeddings | [sentence-transformers](https://www.sbert.net) (`all-MiniLM-L6-v2`) |
| Vector DB | [ChromaDB](https://www.trychroma.com) |
| LLM | [Groq API](https://groq.com) — LLaMA 3 8B |
| RAG Framework | [LangChain](https://www.langchain.com) |

---

## ⚙️ Configuration

| Parameter | Default | Description |
|---|---|---|
| `chunk_size` | `500` words | Size of each document chunk |
| `overlap` | `50` words | Overlap between consecutive chunks |
| `top_k` | `3` | Number of chunks retrieved per query |
| LLM Model | `llama3-8b-8192` | Groq model used for generation |

These can be adjusted directly in `rag_engine.py`.

---

##  API Key Security

Your Groq API key is entered at runtime and stored only as an environment variable for the session — it is **never saved to disk or committed to the repository**.

---

##  Requirements

- Python 3.9+
- A free [Groq API key](https://console.groq.com)

---

## Live Demo
 [Click here to try it](https://rag-chatbot-7qxz8mcoxwsdqyqh9wvdeq.streamlit.app/)

---

##  License

This project is open source. Feel free to fork, modify, and build on it.
