import streamlit as st
import os
import tempfile
from rag_engine import ingest_document, query_documents

st.set_page_config(page_title="RAG Chatbot 🤖", layout="wide")
st.title("RAG Chatbot 🤖")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input(
        "Groq API Key",
        type="password",
        help="Get your free key at console.groq.com",
    )
    if api_key:
        os.environ["GROQ_API_KEY"] = api_key
        st.success("API Key set ✅")

    st.divider()

    st.header("📄 Upload Document")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file is not None:
        if st.button("Ingest Document"):
            if not api_key:
                st.error("Please enter your Groq API key first.")
            else:
                with st.spinner("Ingesting document…"):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name
                    try:
                        num_chunks = ingest_document(tmp_path, uploaded_file.name)
                        st.success(f"✅ {uploaded_file.name} → {num_chunks} chunks stored.")
                    except Exception as e:
                        st.error(f"Ingestion failed: {e}")
                    finally:
                        os.unlink(tmp_path)

# ── Chat ──────────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question about your document…"):
    if not api_key:
        st.warning("Please enter your Groq API key in the sidebar.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    answer, source_chunks = query_documents(prompt)
                    st.markdown(answer)
                    with st.expander("📚 Source chunks used"):
                        for i, chunk in enumerate(source_chunks, 1):
                            st.markdown(f"**Chunk {i}:** {chunk}")
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except ValueError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f"Error: {e}")
