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
