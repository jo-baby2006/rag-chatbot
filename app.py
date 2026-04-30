import os
import tempfile
import streamlit as st
from rag_engine import ingest_document, list_ingested_sources, query_documents

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")
st.title("RAG Chatbot 🤖")
st.caption("Upload PDF documents and ask questions about them.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "api_key_set" not in st.session_state:
    st.session_state.api_key_set = False

with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Groq API Key", type="password", help="Get your free key at console.groq.com", placeholder="gsk_...")
    if api_key:
        os.environ["GROQ_API_KEY"] = api_key
        st.session_state.api_key_set = True
        st.success("✅ API Key set!")
    elif not st.session_state.api_key_set:
        st.warning("Please enter your Groq API key to enable answering.")

    st.divider()
    st.header("📄 Upload Documents")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"], label_visibility="collapsed")

    if uploaded_file is not None:
        doc_name = uploaded_file.name.replace(" ", "_").replace(".pdf", "")
        if st.button("📥 Ingest Document", use_container_width=True):
            with st.spinner(f"Ingesting '{uploaded_file.name}'…"):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name
                    num_chunks = ingest_document(tmp_path, doc_name)
                    os.unlink(tmp_path)
                    st.success(f"✅ Ingested {num_chunks} chunks from '{uploaded_file.name}'.")
                except ValueError as exc:
                    st.error(f"❌ {exc}")
                except Exception as exc:
                    st.error(f"❌ Unexpected error: {exc}")

    st.divider()
    st.header("📚 Ingested Documents")
    sources = list_ingested_sources()
    if sources:
        for src in sources:
            st.markdown(f"- `{src}`")
    else:
        st.info("No documents ingested yet.")

    st.divider()
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📎 Source chunks used"):
                for i, chunk in enumerate(msg["sources"], 1):
                    st.markdown(f"**Chunk {i}:** {chunk[:400]}…")

user_question = st.chat_input("Ask a question about your documents…", disabled=not st.session_state.api_key_set)

if user_question:
    st.session_state.chat_history.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                answer, source_chunks = query_documents(user_question)
            except Exception as exc:
                answer = f"❌ Error: {exc}"
                source_chunks = []
        st.markdown(answer)
        if source_chunks:
            with st.expander("📎 Source chunks used"):
                for i, chunk in enumerate(source_chunks, 1):
                    st.markdown(f"**Chunk {i}:** {chunk[:400]}…")

    st.session_state.chat_history.append({"role": "assistant", "content": answer, "sources": source_chunks})
