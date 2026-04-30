import streamlit as st
import os
import tempfile
from rag_engine import ingest_document, query_documents

st.set_page_config(page_title="RAG Chatbot 🤖", layout="centered")
st.title("RAG Chatbot 🤖")

# ── API Key ──────────────────────────────────────────────────────────────────
api_key = st.text_input(
    "Groq API Key",
    type="password",
    help="Get your free key at console.groq.com",
)

if api_key:
    os.environ["GROQ_API_KEY"] = api_key
    st.success("API Key set successfully!")
else:
    st.warning("Please enter your Groq API key to continue.")
    st.stop()          # Don't render the rest until the key is provided

# ── PDF Upload & Ingestion ────────────────────────────────────────────────────
st.subheader("📄 Upload a PDF Document")
uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file is not None:
    doc_name = uploaded_file.name.replace(".pdf", "").replace(" ", "_")

    if st.button("Ingest Document"):
        with st.spinner("Ingesting document..."):
            # Save upload to a temp file so fitz can open it
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            try:
                num_chunks = ingest_document(tmp_path, doc_name)
                st.success(f"✅ Ingested **{num_chunks}** chunks from '{uploaded_file.name}'.")
                st.session_state["doc_ingested"] = True
            except Exception as e:
                st.error(f"❌ Ingestion failed: {e}")
            finally:
                os.unlink(tmp_path)   # Clean up temp file

# ── Q&A Section ───────────────────────────────────────────────────────────────
st.subheader("💬 Ask a Question")

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

user_question = st.text_input("Ask a question about your document:")

if user_question:
    with st.spinner("Searching document and generating answer..."):
        try:
            answer, context_chunks = query_documents(user_question)
            st.session_state["chat_history"].append(
                {"question": user_question, "answer": answer, "chunks": context_chunks}
            )
        except ValueError as e:
            st.error(f"❌ {e}")
        except Exception as e:
            st.error(f"❌ Error: {e}")

# Display chat history (most recent first)
for entry in reversed(st.session_state["chat_history"]):
    st.markdown(f"**🙋 You:** {entry['question']}")
    st.markdown(f"**🤖 Answer:** {entry['answer']}")
    with st.expander("📚 Source chunks used"):
        for i, chunk in enumerate(entry["chunks"], 1):
            st.markdown(f"**Chunk {i}:** {chunk}")
    st.divider()
