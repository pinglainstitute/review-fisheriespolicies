import streamlit as st
import os
from google.oauth2 import service_account
from google.ai import generativelanguage_v1 as glm  # 使用 v1 模块
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from dotenv import load_dotenv

# 设置 Streamlit 页面配置
# Set up the Streamlit page configuration
st.set_page_config(
    page_title="Chat PDF",
    page_icon="💁",
    layout="wide",
)

# 加载环境变量
# Load environment variable
load_dotenv(dotenv_path=".env")
SERVICE_ACCOUNT_FILE = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE")

# 初始化 Google API 客户端
# Initialize the Google API client
def init_generative_language_api(service_account_file):
    try:
        credentials = service_account.Credentials.from_service_account_file(service_account_file)
        client = glm.GenerativeServiceClient(credentials=credentials)
        return client
    except Exception as e:
        st.error(f"Error initializing Google API client: {e}")
        st.stop()


def generate_content(client, model, prompt):
    try:
        request = glm.GenerateContentRequest(
            model=model,
            contents=[{"parts": [{"text": prompt}]}]
        )
        response = client.generate_content(request)

        # 直接处理响应的文本内容
        # Process the text content of the response directly
        if response.candidates:
            return response.candidates[0].content.parts[0].text
        else:
            st.error("No content generated.")
            return None
    except Exception as e:
        st.error(f"Error generating content: {e}")
        return None


# PDF 文件处理
# PDF file processing
def get_pdf_text(pdf_docs):
    text = ""
    if isinstance(pdf_docs, list):  # multiple documents
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
    else:  # single document
        pdf_reader = PdfReader(pdf_docs)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)  
    chunks = text_splitter.split_text(text)
    return chunks



def main():
    # 初始化 Google API 客户端
    # Initialize the Google API client
    SERVICE_ACCOUNT_FILE = "./enduring-lane-443604-p5-ccb4cf47abd9.json"

    st.write("Loaded Environment Variable:", SERVICE_ACCOUNT_FILE)

    client = init_generative_language_api(SERVICE_ACCOUNT_FILE)

    st.header("Chat with PDF using Generative Language API 💁")

    # 初始化 session state 来存储上下文
    # Initialize the session state to store the context
    if 'pdf_text' not in st.session_state:
        st.session_state.pdf_text = ""

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        # 如果已经上传并处理了PDF
        # If the PDF has been uploaded and processed
        if st.session_state.pdf_text:
            # 构建包含PDF上下文的完整提示
            # Build complete tips that include PDF context
            full_prompt = f"Context from PDF: {st.session_state.pdf_text}\n\nQuestion: {user_question}"
        else:
            full_prompt = user_question

        model = "models/gemini-1.5-flash"
        response = generate_content(client, model, full_prompt)
        if response:
            st.write("Generated Response:")
            st.write(response)
        else:
            st.error("Failed to generate content.")

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button",
                                    accept_multiple_files=True)
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    if raw_text:
                        # 将提取的文本保存到 session state
                        # Saves the extracted text to the session state
                        st.session_state.pdf_text = raw_text

                        text_chunks = get_text_chunks(raw_text)
                        st.success("Done processing PDF files.")
                        
                        st.write("Extracted Text:")
                        # The first 1000 characters are displayed
                        st.write(raw_text[:1000])  # 显示前1000字符
                    else:
                        st.error("No text found in the uploaded PDFs.")
            else:
                st.error("Please upload at least one PDF file.")


if __name__ == "__main__":
    main()
