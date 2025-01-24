import streamlit as st
import os
from google.oauth2 import service_account
from google.ai import generativelanguage_v1 as glm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from dotenv import load_dotenv

# Set up Streamlit page configuration
st.set_page_config(
    page_title="Chat PDF & Symbolic Reasoning",
    page_icon="💁",
    layout="wide",
)

# Symbol class for symbolic reasoning
class Symbol:
    def __init__(self, statement, only_nesy=False):
        self.statement = statement
        self.only_nesy = only_nesy

    def __and__(self, other):
        return Symbol(f"({self.statement} AND {other.statement})", self.only_nesy)

    def __or__(self, other):
        return Symbol(f"({self.statement} OR {other.statement})", self.only_nesy)

    def extract(self, context):
        if context == 'answer':
            return f"Based on: {self.statement}"

# Symbolic reasoning interface
def symbolic_reasoning_interface():
    st.subheader("Symbolic Reasoning Tool")
    symbol_1 = st.text_input("Enter the first symbolic statement:")
    symbol_2 = st.text_input("Enter the second symbolic statement:")

    if st.button("Perform Symbolic Reasoning"):
        if symbol_1 and symbol_2:
            S1, S2 = Symbol(symbol_1, only_nesy=True), Symbol(symbol_2)
            result = S1 & S2
            st.write("Logical Reasoning Result:", result.extract('answer'))
        else:
            st.error("Please provide both symbolic statements.")

# Initialize the Google API client
def init_generative_language_api(service_account_file):
    try:
        credentials = service_account.Credentials.from_service_account_file(service_account_file)
        return glm.GenerativeServiceClient(credentials=credentials)
    except Exception as e:
        st.error(f"Error initializing Google API client: {e}")
        st.stop()

# Generate content using Google API
def generate_content(client, model, prompt):
    try:
        request = glm.GenerateContentRequest(
            model=model,
            contents=[{"parts": [{"text": prompt}]}]
        )
        response = client.generate_content(request)
        return response.candidates[0].content.parts[0].text if response.candidates else None
    except Exception as e:
        st.error(f"Error generating content: {e}")
        return None

# Extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

# Split text into chunks
def get_text_chunks(text, chunk_size=10000, overlap=1000):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_text(text)

# Main application
def main():
    # Load environment variables
    load_dotenv(dotenv_path=".env")
    service_account_file = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE", "./enduring-lane-443604-p5-ccb4cf47abd9.json")

    # Initialize Google API client
    client = init_generative_language_api(service_account_file)

    st.header("Chat PDF & Symbolic Reasoning 💁")

    # Sidebar menu
    with st.sidebar:
        st.title("Menu:")
        app_mode = st.radio("Choose Mode:", ["Chat PDF", "Symbolic Reasoning"])

    if app_mode == "Chat PDF":
        # PDF chat implementation
        st.subheader("Chat with PDF")
        pdf_docs = st.file_uploader("Upload PDF Files", accept_multiple_files=True)
        if st.button("Process PDFs"):
            if pdf_docs:
                with st.spinner("Processing PDFs..."):
                    raw_text = get_pdf_text(pdf_docs)
                    if raw_text:
                        st.session_state["pdf_text"] = raw_text
                        st.success("PDF processing complete.")
                        st.write("Extracted Text (first 1000 characters):")
                        st.write(raw_text[:1000])
                    else:
                        st.error("No text found in the uploaded PDFs.")
            else:
                st.error("Please upload at least one PDF file.")

        user_question = st.text_input("Ask a Question from the PDF Files")
        if user_question and "pdf_text" in st.session_state:
            context = f"Context from PDF: {st.session_state['pdf_text']}\n\nQuestion: {user_question}"
            model = "models/gemini-1.5-flash"
            response = generate_content(client, model, context)
            if response:
                st.write("Generated Response:", response)
            else:
                st.error("Failed to generate content.")

    elif app_mode == "Symbolic Reasoning":
        # Symbolic reasoning implementation
        symbolic_reasoning_interface()

if __name__ == "__main__":
    main()