import streamlit as st
import os
import asyncio
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from pyreason import KnowledgeBase, Reasoner

# 设置 Streamlit 页面配置
st.set_page_config(
    page_title="Chat PDF with Reasoning",
    page_icon="🤖",
    layout="wide",
)

# 加载环境变量 🌟
@st.cache_resource
def load_env():
    return "./enduring-lane-443604-p5-2b4c36a58551.json"


# 初始化 Google API 客户端（延迟加载）
@st.cache_resource
def init_generative_language_api(service_account_file):
    from google.oauth2 import service_account
    from google.ai import generativelanguage_v1 as glm

    try:
        credentials = service_account.Credentials.from_service_account_file(service_account_file)
        client = glm.GenerativeServiceClient(credentials=credentials)
        return client
    except Exception as e:
        st.error(f"Error initializing Google API client: {e}")
        st.stop()


# 异步生成内容（优化为异步任务）
async def generate_content_async(client, model, prompt):
    loop = asyncio.get_event_loop()
    from google.ai import generativelanguage_v1 as glm

    def generate_request():
        request = glm.GenerateContentRequest(
            model=model,
            contents=[{"parts": [{"text": prompt}]}]
        )
        return client.generate_content(request)

    response = await loop.run_in_executor(None, generate_request)
    if response.candidates:
        return response.candidates[0].content.parts[0].text
    else:
        return "No content generated."


# PDF 文件处理
@st.cache_data
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text


@st.cache_data
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)  # 减小块大小提高效率
    return text_splitter.split_text(text)


# 初始化 PyReason 知识库和推理引擎（延迟加载）
@st.cache_resource
def init_pyreason():
    from pyreason import KnowledgeBase, Reasoner

    kb = KnowledgeBase()
    kb.add_rule("if question_contains('logic') and pdf_context_is('absent') then recommend('provide more context')")
    kb.add_rule("if answer_is('unclear') and pdf_context_is('present') then recommend('clarify the response')")

    reasoner = Reasoner(kb)
    return reasoner


# 验证答案合理性
def validate_with_pyreason(reasoner, question, pdf_context, answer):
    reasoning_input = {
        "question": question,
        "pdf_context": "present" if pdf_context else "absent",
        "answer": "unclear" if not answer else "clear"
    }
    return reasoner.infer(reasoning_input)


# 主函数
def main():
    # 加载环境变量和 Google API 客户端
    SERVICE_ACCOUNT_FILE = load_env()

    if not SERVICE_ACCOUNT_FILE:
        st.error("Service account file path is not set.")
        st.stop()
    client = init_generative_language_api(SERVICE_ACCOUNT_FILE)

    st.header("Chat with PDF using Generative Language API and Logical Reasoning 🤖")

    if 'pdf_text' not in st.session_state:
        st.session_state.pdf_text = ""

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        # 构建提示（优化：减少上下文长度）
        pdf_context = st.session_state.pdf_text
        full_prompt = f"Keywords: {pdf_context[:300]}\n\nQuestion: {user_question}" if pdf_context else user_question

        # 异步调用生成内容（优化为异步任务）
        model = "models/gemini-1.5-flash"
        with st.spinner("Generating response..."):
            response = asyncio.run(generate_content_async(client, model, full_prompt))

        # 初始化 PyReason 并验证答案
        reasoner = init_pyreason()
        reasoning_result = validate_with_pyreason(reasoner, user_question, pdf_context, response)

        # 根据验证结果调整回复（优化：限制输出长度）
        if reasoning_result.get("recommend") == "provide more context":
            st.warning("Please upload a PDF or provide more context.")
        elif reasoning_result.get("recommend") == "clarify the response":
            refined_response = asyncio.run(generate_content_async(client, model, f"Clarify: {response}"))
            st.write("Refined Response (50 chars):")
            st.write(refined_response[:50])  # 限制输出为50字
        else:
            st.write("Generated Response (50 chars):")
            st.write(response[:50])  # 限制输出为50字

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button",
                                    accept_multiple_files=True)
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    if raw_text:
                        st.session_state.pdf_text = raw_text
                        st.success("Done processing PDF files.")
                        st.write("Extracted Text:")
                        st.write(raw_text[:500])  # 显示前500字符
                    else:
                        st.error("No text found in the uploaded PDFs.")
            else:
                st.error("Please upload at least one PDF file.")


if __name__ == "__main__":
    main()