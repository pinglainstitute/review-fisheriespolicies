# llm2_updated.py
import os
import streamlit as st

# --- deps 更新：用 pypdf 替代 PyPDF2，FAISS 从社区包导入 ---
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from dotenv import load_dotenv
load_dotenv()

# Google Generative AI / LangChain 集成
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)
import google.generativeai as genai

# FAISS 改为社区包路径（0.1+后分包）
from langchain_community.vectorstores import FAISS

from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

from google import genai
from google.genai import types

# ---------- 初始化 ----------
client = genai.Client()

API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    st.error("API key is missing. Please set GOOGLE_API_KEY in your .env or environment.")
else:
    genai.configure(api_key=API_KEY)

INDEX_DIR = "faiss_index"  # 持久化路径
EMBED_MODEL = "text-embedding-004"  # 建议使用最新版；旧值 models/embedding-001 已过时


# ---------- 工具函数 ----------
def get_pdf_text(pdf_docs):
    """将上传的多个 PDF 合并为纯文本。"""
    text = []
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            page_text = page.extract_text() or ""
            if page_text.strip():
                text.append(page_text)
    return "\n".join(text)


def get_text_chunks(text):
    """分块策略：较小块+部分重叠，提升召回与上下文连贯。"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        length_function=len,
    )
    return splitter.split_text(text)


@st.cache_resource(show_spinner=False)
def build_or_load_vector_store(_chunks=None):
    """
    - 如果已有本地索引：加载
    - 否则：用传入的 chunks 构建并保存
    """
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)

    # 先尝试加载
    if os.path.isdir(INDEX_DIR):
        try:
            # 不使用 allow_dangerous_deserialization，避免老版本 pickle 风险
            return FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=False)
        except Exception:
            # 索引格式不兼容时，重建
            pass

    # 构建新索引
    if not _chunks:
        raise ValueError("No chunks provided to build a new FAISS index.")
    vs = FAISS.from_texts(_chunks, embedding=embeddings)
    vs.save_local(INDEX_DIR)
    return vs


def get_conversational_chain():
    """最小改动沿用 load_qa_chain（stuff），后续可逐步迁移到 LCEL。"""
    prompt_template = """
Understand the question and answer strictly based on the provided context.
Write a step-by-step, precise, and well-structured answer.

Context:
{context}

Question:
{question}

Answer:
    """.strip()

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro-002",
        temperature=0.2,  # 更稳定的回答；之前 1.2 偏发散
    )
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
    return chain


# def answer_question(vs, user_question: str):
#     """相似度检索 + QA 链"""
#     docs = vs.similarity_search(user_question, k=4)
#     chain = get_conversational_chain()
#     result = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
#     return result.get("output_text", "").strip() or "No output produced."
def answer_question(vs, user_question: str):
    """相似度检索 + Gemini 2.5 Flash (Thinking)"""
    docs = vs.similarity_search(user_question, k=4)
    context_text = "\n\n".join([d.page_content for d in docs])

    # 自定义 prompt
    prompt = f"""
    You are an expert fisheries policy assistant. 
    Answer the question using ONLY the following context. 
    Be precise and structured. If not found, say "I don't know."

    Context:
    {context_text}

    Question:
    {user_question}
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=50)  # 💡 开启思考
        ),
    )
    return response.text.strip() if response.text else "No output produced."


# ---------- Streamlit UI ----------
def main():
    st.set_page_config(page_title="AI Fisheries Manager 🐟", page_icon="🐟")
    st.title("AI Fisheries Manager 🐟")
    st.image("https://pingla.org.au/images/Pingala_Logo_option_7.png", use_container_width=False)

    # 侧边栏：上传与建索引
    with st.sidebar:
        st.header("Documents")
        pdf_docs = st.file_uploader(
            "Upload PDF files then click 'Submit & Process'",
            type=["pdf"],
            accept_multiple_files=True,
        )
        if st.button("Submit & Process", type="primary", help="Extract, chunk, and index PDFs"):
            if not pdf_docs:
                st.error("Please upload at least one PDF.")
            else:
                with st.spinner("Extracting & indexing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    if not raw_text.strip():
                        st.error("No extractable text found in the PDFs.")
                    else:
                        chunks = get_text_chunks(raw_text)
                        # 强制重建索引：先清掉旧目录（避免不同版本残留）
                        if os.path.isdir(INDEX_DIR):
                            import shutil
                            shutil.rmtree(INDEX_DIR, ignore_errors=True)
                        _ = build_or_load_vector_store(chunks)  # 构建并缓存
                        st.success("Index built successfully ✅")

        st.divider()
        if os.path.isdir(INDEX_DIR):
            st.caption("FAISS index detected ✔︎")
        else:
            st.caption("No index yet. Please upload PDFs and build the index.")

    # 主区：问答
    user_q = st.text_input("Ask the fisheries manager a question")
    if user_q:
        if not API_KEY:
            st.error("Missing GOOGLE_API_KEY.")
            return
        if not os.path.isdir(INDEX_DIR):
            st.warning("No index found. Please upload PDFs and click 'Submit & Process' first.")
            return
        with st.spinner("Retrieving & answering..."):
            vs = build_or_load_vector_store()
            answer = answer_question(vs, user_q)
        st.markdown("**Reply:**")
        st.write(answer)


if __name__ == "__main__":
    main()
