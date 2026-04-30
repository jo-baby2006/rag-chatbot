api_key = st.text_input(
    "Gemini API Key",
    type="password",
    help="Get your free key at aistudio.google.com"
)
if api_key:
    os.environ["GEMINI_API_KEY"] = api_key
