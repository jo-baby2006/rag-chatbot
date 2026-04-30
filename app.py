import streamlit as st
import os

st.title("RAG Chatbot 🤖")

api_key = st.text_input(
    "Groq API Key",
    type="password",
    help="Get your free key at console.groq.com"
)

if api_key:
    os.environ["GROQ_API_KEY"] = api_key
    st.success("API Key set successfully!")

    # 👇 Add user question input
    user_question = st.text_input("Ask your question:")

    if user_question:
        # Temporary response (we will connect backend next)
        st.write("You asked:", user_question)
        st.write("Answer: Coming soon...")
