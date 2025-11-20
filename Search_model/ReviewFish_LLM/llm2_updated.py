# llm2_updated.py
import os
import shutil
import streamlit as st

from dotenv import load_dotenv
load_dotenv()

# ---------- Keys & Client ----------
# # 统一读取 KEY；LangChain 的 Google embeddings 习惯读 GOOGLE_API_KEY
# API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
# if not API_KEY:
#     st.error("API key is missing. Set GEMINI_API_KEY or GOOGLE_API_KEY in .env / env.")
#     st.stop()

# # 确保 LangChain 能读到（需要 GOOGLE_API_KEY 这个变量名）
# os.environ["GOOGLE_API_KEY"] = API_KEY

# # 新版 Google GenAI 客户端（不再使用 genai.configure）
# from google import genai
# from google.genai import types
#client = genai.Client(api_key=API_KEY)
# this version is modefied on 19 Novs
# ---------- Keys & Client ----------
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

# ---------- Helpers ----------
# def get_docs_with_meta(pdf_files):
#     """读取多个 PDF，逐页提取为 Document，并保留来源与页码。"""
#     docs = []
#     for f in pdf_files:
#         reader = PdfReader(f)
#         name = getattr(f, "name", "uploaded.pdf")
#         for i, page in enumerate(reader.pages):
#             txt = (page.extract_text() or "").strip()
#             if not txt:
#                 continue
#             docs.append(
#                 Document(
#                     page_content=txt,
#                     metadata={"source": name, "page": i + 1}
#                 )
#             )
#     return docs
# ---------- Helpers ---------- optimize
def get_docs_with_meta(pdf_files):
    """读取多个 PDF，逐页提取为 Document，并保留来源与页码；增加文本清洗。"""
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

@st.cache_resource(show_spinner=False)
def build_or_load_vector_store(_documents=None):
    """
    若本地已有索引则加载；否则用传入的 documents 构建并保存。
    """
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)

    if os.path.isdir(INDEX_DIR):
        try:
            return FAISS.load_local(
                INDEX_DIR, embeddings, allow_dangerous_deserialization=False
            )
        except Exception:
            pass  # 版本或格式不兼容，走重建

    if not _documents:
        raise ValueError("No documents provided to build a new FAISS index.")

    vs = FAISS.from_documents(_documents, embedding=embeddings)
    vs.save_local(INDEX_DIR)
    return vs


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
    """更稳健的 MMR 检索逻辑，清晰分层。 similarity_with_score, fallback"""
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

    """更稳健的检索 + 结构化回答LLM生成 + 引用标注"""

    docs = robust_retrieve(vs, user_question, k=10)
    if not docs:
        # return "I don't know.", []
        st.warning("⚠️ No documents retrieved for this query.")

    # # 1) 用 MMR，先抓更多候选（fetch_k），再去冗余
    # try:
    #     docs = vs.max_marginal_relevance_search(
    #         user_question, k=10, fetch_k=40, lambda_mult=0.2
    #     )
    # except Exception:
    #     # 兼容没有 MMR 的索引
    #     docs = vs.similarity_search(user_question, k=12)

    #---------------------------------version1-----------------------------------#
    '''
    try:
        pairs = vs.similarity_search_with_score(user_question, k=30)
        # LangChain-FAISS 的得分含义随索引类型不同，这里仅做排序与再筛
        pairs = sorted(pairs, key=lambda x: x[1])  # 分数小=更近（常见情形）
        docs = [d for d, _ in pairs[:12]]
        # 再跑一轮 MMR 去冗余
        docs = vs.max_marginal_relevance_search(user_question, k=10, fetch_k=30, lambda_mult=0.2)
    except Exception:
        docs = vs.max_marginal_relevance_search(user_question, k=10, fetch_k=40, lambda_mult=0.2)
    '''

    #---------------------------------version2-----------------------------------#
    


    # 3) 把来源页码嵌进上下文，鼓励模型引用
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

    # 4) 更“进取”的提示：有证据就作答，并在句尾打 [S#] 引用
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

    # resp = client.models.generate_content(
    #     model="gemini-2.5-flash",
    #     contents=prompt,
    #     config=types.GenerateContentConfig(
    #         thinking_config=types.ThinkingConfig(thinking_budget=20)
    #     ),
    # ✅ 创建模型实例
    model = genai.GenerativeModel("gemini-2.5-flash")

    # ✅ 调用 generate_content
    resp = model.generate_content(
        prompt,
        generation_config={
            "temperature": 0.2,
            "max_output_tokens": 1024,
        },
    )
    # 提取回答文本
    answer = (getattr(resp, "text", "") or "").strip()
    if not answer:
        answer = "I don't know."
 
    
    # ✅ 在这里加“最近片段提示”逻辑
    if answer.strip() == "I don't know." and used:
        hints = [f"• {s or 'PDF'} p.{p}" for s,p in used[:3]]
        st.info("Closest sections:\n" + "\n".join(hints))

    return answer, used

@st.cache_resource(show_spinner=False)
def build_or_load_vector_store(_documents=None):
    """
    优先加载已有索引；否则在提供了 documents 时构建。
    若既没有索引、也没传入 documents，则返回 None（不抛异常）。
    """
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)

    # 尝试加载已有索引
    if os.path.isdir(INDEX_DIR):
        try:
            return FAISS.load_local(
                INDEX_DIR, embeddings, allow_dangerous_deserialization=False
            )
        except Exception:
            # 版本或格式不兼容，继续尝试重建
            pass

    # 没有本地索引；若提供了文档则构建，否则返回 None
    if _documents:
        vs = FAISS.from_documents(_documents, embedding=embeddings)
        vs.save_local(INDEX_DIR)
        return vs

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
                    raw_docs = get_docs_with_meta(pdf_docs)
                    if not raw_docs:
                        st.error("No extractable text found in the PDFs.")
                    else:
                        splitter = RecursiveCharacterTextSplitter(
                            # chunk_size=2000, chunk_overlap=200 ## optimize
                            chunk_size=2500,   # 适度放大
                            chunk_overlap=300, # 保留跨段语义
                            length_function=len,
                        )
                        chunks = splitter.split_documents(raw_docs)

                        # 重建索引：先清掉旧目录
                        if os.path.isdir(INDEX_DIR):
                            shutil.rmtree(INDEX_DIR, ignore_errors=True)

                        # _ = build_or_load_vector_store(chunks)  #optimize
                        # st.success("Index built successfully ✅")
                        # 提交后构建
                        vs = build_or_load_vector_store(chunks)
                        if vs is None:
                            st.error("Index build failed. Please re-upload PDFs and click 'Submit & Process' again.")
                        else:
                            st.success("Index built successfully ✅")

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