# llm2_updated.py
import os
import shutil
import streamlit as st

from dotenv import load_dotenv
load_dotenv()

# ---------- 配置 API 密钥 ----------
API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not API_KEY:
    st.error("API key is missing. Set GEMINI_API_KEY or GOOGLE_API_KEY in .env / env.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = API_KEY

import google.generativeai as genai
from google.generativeai import types

genai.configure(api_key=API_KEY)




# ---------- LangChain / PDF / Vector store ----------
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

#（如果你想切换到 LangChain 的对话链，保留这两个；当前示例直接走 client 调用）
# from langchain.chains.question_answering import load_qa_chain
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.prompts import PromptTemplate

INDEX_DIR = "faiss_index"
EMBED_MODEL = "models/text-embedding-004"   # 注意带 models/ 前缀

# ---------- 辅助函数 ----------
def get_docs_with_meta(pdf_files):
    """Read multiple PDFs, extract page by page as Documents, preserve source and page numbers; with text cleaning."""
    docs = []
    for f in pdf_files:
        reader = PdfReader(f)
        name = getattr(f, "name", "uploaded.pdf")

        for i, page in enumerate(reader.pages):
            # 提取 + 清洗文本
            raw_txt = page.extract_text() or ""
            # 去掉多余空白与换行，统一成一行
            txt = " ".join(raw_txt.split())

            # 极短内容一般是页眉页脚/空页，直接跳过
            if len(txt) < 40:
                continue

            docs.append(
                Document(
                    page_content=txt,
                    metadata={"source": name, "page": i + 1}
                )
            )
    return docs

# def answer_question(vs, user_question: str):
#     """相似度检索 + Gemini 2.5 Flash（可选 Thinking）"""
#     docs = vs.similarity_search(user_question, k=8) #improve
#     context_text = "\n\n".join([d.page_content for d in docs])

#     prompt = f"""
# You are an expert fisheries policy assistant.
# Answer the question using ONLY the following context.
# Be precise and structured. If not found, say "I don't know."

# Context:
# {context_text}

# Question:
# {user_question}
# """.strip()

#     resp = client.models.generate_content(
#         model="gemini-2.5-flash",
#         contents=prompt,
#         config=types.GenerateContentConfig(
#             thinking_config=types.ThinkingConfig(thinking_budget=30)  # 低一些以提速/省费
#         ),
#     )
#     answer = (resp.text or "").strip() if hasattr(resp, "text") else ""
#     if not answer:
#         answer = "No output produced."

#     # 返回答案与来源，方便 UI 展示
#     sources = [(d.metadata.get("source"), d.metadata.get("page")) for d in docs]
#     return answer, sources
def robust_retrieve(vs, query, k=10):
    """More robust MMR retrieval logic with clear layering. similarity_with_score, fallback"""
    try:
        # 先拿初步候选 （按相似度得分排序）
        pairs = vs.similarity_search_with_score(query, k=30)
        pairs = sorted(pairs, key=lambda x: x[1])
        top_docs = [d for d, _ in pairs[:12]]

        # 再跑一轮 MMR 去重，用MMR去冗余
        mmr_docs = vs.max_marginal_relevance_search(
            query, k=k, fetch_k=30, lambda_mult=0.2
        )
        return mmr_docs or top_docs

    except Exception as e:
        # fallback: 简单 similarity_search
        st.warning(f"MMR retrieval failed, falling back to similarity_search. ({e})")
        return vs.similarity_search(query, k=k)
    
def answer_question(vs, user_question: str):

    """More robust retrieval + structured LLM answer generation + citation markup"""

    docs = robust_retrieve(vs, user_question, k=10)
    if not docs:
        # return "I don't know.", []
        st.warning("⚠️ No documents retrieved for this query.")

    # 构建上下文并加入来源信息
    blocks, used = [], []
    MAX_CTX = 12000  # 防止过长被截断
    total = 0
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "PDF")
        pg  = d.metadata.get("page", "?")
        block = f"[S{i} | {src} p.{pg}]\n{d.page_content.strip()}"
        if total + len(block) > MAX_CTX:
            break
        blocks.append(block)
        used.append((src, pg))
        total += len(block)

    context_text = "\n\n".join(blocks)

    # 构建提示词
    prompt = f"""
You are an expert fisheries policy assistant.
Use ONLY the context below. If the context contains relevant facts, answer concisely and cite sources like [S1], [S2] at the end of sentences derived from them.
Only say "I don't know." if the context truly does not contain the answer.

Context:
{context_text}

Question:
{user_question}

Answer (with citations):
""".strip()

    # 调用 Gemini 模型生成答案
    model = genai.GenerativeModel("gemini-2.5-flash")
    resp = model.generate_content(
        prompt,
        generation_config={
            "temperature": 1.1,
            "max_output_tokens": 4096,  # 增加输出长度限制，避免答案被截断
        },
    )
    # 提取回答文本
    answer = (getattr(resp, "text", "") or "").strip()
    if not answer:
        answer = "I don't know."
    
    # 如果没有找到答案，显示最近的相关片段
    if answer.strip() == "I don't know." and used:
        hints = [f"• {s or 'PDF'} p.{p}" for s,p in used[:3]]
        st.info("Closest sections:\n" + "\n".join(hints))

    return answer, used

@st.cache_resource(show_spinner=False)
def build_or_load_vector_store(_documents=None):
    """
    Load existing index if available; otherwise build when documents are provided.
    Returns None if no index exists and no documents are provided (no exception thrown).
    """
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)
    except Exception as e:
        st.error(f"❌ Embeddings initialization failed: {str(e)}")
        return None

    # 尝试加载已有索引
    if os.path.isdir(INDEX_DIR):
        try:
            return FAISS.load_local(
                INDEX_DIR, embeddings, allow_dangerous_deserialization=False
            )
        except Exception as e:
            # 版本或格式不兼容，继续尝试重建
            st.warning(f"⚠️ Failed to load existing index, will rebuild: {str(e)}")
            pass

    # 没有本地索引；若提供了文档则构建，否则返回 None
    if _documents:
        try:
            st.info(f"📊 Generating vector index for {len(_documents)} document chunks...")
            vs = FAISS.from_documents(_documents, embedding=embeddings)
            vs.save_local(INDEX_DIR)
            return vs
        except Exception as e:
            st.error(f"❌ Index building failed: {str(e)}")
            return None

    # 既无索引，也没文档 —— 交给上层友好提示
    return None

# ---------- Streamlit UI ----------
def main():
    st.set_page_config(page_title="AI Fisheries Manager 🐟", page_icon="🐟")
    st.title("AI Fisheries Manager 🐟")
    st.image("https://pingla.org.au/images/Pingala_Logo_option_7.png", width=300)

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
                    try:
                        # 提取文档
                        st.info(f"📄 Extracting {len(pdf_docs)} PDF file(s)...")
                        raw_docs = get_docs_with_meta(pdf_docs)
                        if not raw_docs:
                            st.error("❌ Failed to extract text content from PDFs.")
                        else:
                            st.success(f"✅ Successfully extracted {len(raw_docs)} page(s)")
                            
                            # 分割文档
                            splitter = RecursiveCharacterTextSplitter(
                                chunk_size=2500,   # 适度放大
                                chunk_overlap=300, # 保留跨段语义
                                length_function=len,
                            )
                            chunks = splitter.split_documents(raw_docs)
                            st.info(f"📝 Documents split into {len(chunks)} chunk(s)")

                            # 重建索引：先清掉旧目录
                            if os.path.isdir(INDEX_DIR):
                                shutil.rmtree(INDEX_DIR, ignore_errors=True)

                            # 构建向量索引
                            vs = build_or_load_vector_store(chunks)
                            if vs is None:
                                st.error("❌ Index building failed. Please check error messages and retry.")
                            else:
                                st.success("✅ Index built successfully! You can now start asking questions.")
                    except Exception as e:
                        st.error(f"❌ Processing error: {str(e)}")
                        import traceback
                        st.error(f"Detailed error:\n```\n{traceback.format_exc()}\n```")

        st.divider()
        st.caption("FAISS index detected ✔︎" if os.path.isdir(INDEX_DIR) else
                   "No index yet. Please upload PDFs and build the index.")

    # 主区：问答
    user_q = st.text_input("Ask the fisheries manager a question")
    if user_q:
        if not os.path.isdir(INDEX_DIR):
            st.warning("No index found. Please upload PDFs and click 'Submit & Process' first.")
            return
        with st.spinner("Retrieving & answering..."):
            vs = build_or_load_vector_store()  ##optimize 
            if vs is None:
                st.warning("Index not ready. Please upload PDFs and click 'Submit & Process' first.")
                st.stop()
                
            answer, sources = answer_question(vs, user_q)

        st.markdown("**Reply:**")
        st.write(answer)

        if sources:
            src_text = ", ".join([f"{s or 'PDF'} p.{p}" for s, p in sources])
            st.caption(f"Sources: {src_text}")


if __name__ == "__main__":
    main()