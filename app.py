import streamlit as st
import tempfile
import os
from rag_engine import ingest_document, query_documents

st.set_page_config(
    page_title="RAG Document Chatbot",
    page_icon="📄",
    layout="centered"
)

# ---- Header ----
st.title("📄 RAG Document Chatbot")
st.caption("Upload a PDF and ask questions about it — powered by Gemini AI")

# ---- Sidebar ----
with st.sidebar:
    st.header("⚙️ Setup")
    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        help="Get your free key at aistudio.google.com"
    )
    if api_key:
        os.environ["GEMINI_API_KEY"] = api_key
        st.success("API key saved!")

    st.divider()
    st.markdown("### How it works")
    st.markdown("""
    1. Paste your Gemini API key
    2. Upload a PDF file
    3. Wait for processing
    4. Ask any question
    5. Get AI-powered answers
    """)

    st.divider()
    st.markdown("### Free tools used")
    st.markdown("""
    - **Streamlit** — UI
    - **ChromaDB** — vector storage
    - **Sentence Transformers** — embeddings
    - **Gemini 1.5 Flash** — AI answers
    """)

# ---- Main Area ----
if not api_key:
    st.warning("👈 Please add your Gemini API key in the sidebar first")
    st.stop()

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"],
    help="Upload any PDF file to start asking questions"
)

if uploaded_file:
    with st.spinner("⏳ Processing your document... please wait"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        num_chunks = ingest_document(tmp_path, uploaded_file.name)
        os.unlink(tmp_path)

    st.success(f"✅ Done! Document split into {num_chunks} chunks and stored.")

    st.divider()
    st.subheader("💬 Chat with your document")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Show previous messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Chat input
    if question := st.chat_input("Ask anything about your document..."):

        # Show user message
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        # Get and show AI answer
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching document..."):
                try:
                    answer, sources = query_documents(question)
                    st.write(answer)

                    with st.expander("📎 Source chunks used to answer"):
                        for i, chunk in enumerate(sources):
                            st.markdown(f"**Chunk {i+1}:**")
                            st.text_area(
                                label="",
                                value=chunk,
                                height=120,
                                key=f"chunk_{i}_{question}"
                            )
                except Exception as e:
                    st.error(f"Something went wrong: {str(e)}")
                    answer = "Error occurred."

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

else:
    st.info("⬆️ Upload a PDF above to get started")
