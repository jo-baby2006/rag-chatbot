api_key = st.text_input(
    "Groq API Key",
    type="password",
    help="Get your free key at console.groq.com"
)
if api_key:
    os.environ["GROQ_API_KEY"] = api_key
