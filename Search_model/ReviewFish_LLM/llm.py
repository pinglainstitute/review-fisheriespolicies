# llm2_updated.py
import os
import shutil
import streamlit as st

from dotenv import load_dotenv
load_dotenv()

# ---------- Keys & Client ----------
# Load the API key from environment variables.
# LangChain's Google embeddings expect the variable name GOOGLE_API_KEY.
API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not API_KEY:
    st.error("API key is missing. Set GEMINI_API_KEY or GOOGLE_API_KEY in .env / env.")
    st.stop()

# Ensure LangChain can access the key using the expected variable name.
os.environ["GOOGLE_API_KEY"] = API_KEY

# Initialize the new Google GenAI client (the old genai.configure is deprecated).
from google import genai
from google.genai import types
client = genai.Client(api_key=API_KEY)

# ---------- LangChain / PDF / Vector store ----------
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# (Optional) If you want to switch to LangChain’s QA chain, keep these imports.
# The current example directly calls the client instead.
# from langchain.chains.question_answering import load_qa_chain
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.prompts import PromptTemplate

INDEX_DIR = "faiss_index"
EMBED_MODEL = "models/text-embedding-004"   # Note: model path must start with "models/"

# ---------- Helpers ----------
# def get_docs_with_meta(pdf_files):
#     """Read multiple PDFs, extract each page as a Document, and store source/page metadata."""
#     ...

# ---------- Helpers (Optimized) ----------
def get_docs_with_meta(pdf_files):
    """
    Read multiple PDFs and extract each page as a LangChain Document,
    including source filename and page number. Performs extra text cleaning.
    """
    docs = []
    for f in pdf_files:
        reader = PdfReader(f)
        name = getattr(f, "name", "uploaded.pdf")

        for i, page in enumerate(reader.pages):
            # Extract + clean text
            raw_txt = page.extract_text() or ""
            # Normalize whitespace and line breaks into a single line
            txt = " ".join(raw_txt.split())

            # Skip pages with very short content (likely headers/footers or blank pages)
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
    Load an existing FAISS index if available.
    Otherwise, build a new index from the provided documents and save it locally.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)

    if os.path.isdir(INDEX_DIR):
        try:
            return FAISS.load_local(
                INDEX_DIR, embeddings, allow_dangerous_deserialization=False
            )
        except Exception:
            # If the format or version is incompatible, rebuild the index.
            pass

    if not _documents:
        raise ValueError("No documents provided to build a new FAISS index.")

    vs = FAISS.from_documents(_documents, embedding=embeddings)
    vs.save_local(INDEX_DIR)
    return vs


# def answer_question(vs, user_question: str):
#     """Similarity search + Gemini 2.5 Flash (optional Thinking)"""
#     ...

def answer_question(vs, user_question: str):
    """Robust retrieval + structured answer generation + citation formatting."""

    # 1) Use MMR (Maximal Marginal Relevance) to fetch more candidates and reduce redundancy.
    try:
        pairs = vs.similarity_search_with_score(user_question, k=30)
        # Sort by distance score (smaller = closer in typical FAISS setups)
        pairs = sorted(pairs, key=lambda x: x[1])
        docs = [d for d, _ in pairs[:12]]
        # Run another MMR pass to remove near-duplicates
        docs = vs.max_marginal_relevance_search(user_question, k=10, fetch_k=30, lambda_mult=0.2)
    except Exception:
        docs = vs.max_marginal_relevance_search(user_question, k=10, fetch_k=40, lambda_mult=0.2)

    # 2) If nothing was retrieved, return a fallback response.
    if not docs:
        return "I don't know.", []

    # 3) Embed page numbers and source names in the context to encourage citations.
    blocks, used = [], []
    MAX_CTX = 12000  # Prevent context from being truncated.
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

    # 4) Prompt: require citations [S#] after factual claims.
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

    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=20)
        ),
    )
    answer = (getattr(resp, "text", "") or "").strip()
    if not answer:
        answer = "I don't know."
    
    # Provide hint of the closest sections when model can't answer.
    if answer.strip() == "I don't know." and used:
        hints = [f"• {s or 'PDF'} p.{p}" for s,p in used[:3]]
        st.info("Closest sections:\n" + "\n".join(hints))

    return answer, used

@st.cache_resource(show_spinner=False)
def build_or_load_vector_store(_documents=None):
    """
    Prefer loading an existing FAISS index.
    If no index exists and documents are provided, build a new one.
    Returns None if neither is available (instead of raising an error).
    """
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)

    # Try loading the existing index
    if os.path.isdir(INDEX_DIR):
        try:
            return FAISS.load_local(
                INDEX_DIR, embeddings, allow_dangerous_deserialization=False
            )
        except Exception:
            # Incompatible format/version → fallback to rebuild
            pass

    # If no index exists and documents are provided, build one
    if _documents:
        vs = FAISS.from_documents(_documents, embedding=embeddings)
        vs.save_local(INDEX_DIR)
        return vs

    # Neither index nor documents → let the caller handle it gracefully
    return None

# ---------- Streamlit UI ----------
def main():
    st.set_page_config(page_title="AI Fisheries Manager 🐟", page_icon="🐟")
    st.title("AI Fisheries Manager 🐟")
    st.image("https://pingla.org.au/images/Pingala_Logo_option_7.png", width=300)

    # Sidebar: PDF upload & indexing
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
                            chunk_size=2500,   # Slightly larger chunks
                            chunk_overlap=300, # Keep cross-section semantics
                            length_function=len,
                        )
                        chunks = splitter.split_documents(raw_docs)

                        # Rebuild index: clear the old directory first
                        if os.path.isdir(INDEX_DIR):
                            shutil.rmtree(INDEX_DIR, ignore_errors=True)

                        vs = build_or_load_vector_store(chunks)
                        if vs is None:
                            st.error("Index build failed. Please re-upload PDFs and try again.")
                        else:
                            st.success("Index built successfully ✅")

        st.divider()
        st.caption("FAISS index detected ✔︎" if os.path.isdir(INDEX_DIR) else
                   "No index yet. Please upload PDFs and build the index.")

    # Main area: Q&A
    user_q = st.text_input("Ask the fisheries manager a question")
    if user_q:
        if not os.path.isdir(INDEX_DIR):
            st.warning("No index found. Please upload PDFs and click 'Submit & Process' first.")
            return
        with st.spinner("Retrieving & answering..."):
            vs = build_or_load_vector_store()
            if vs is None:
                st.warning("Index not ready. Please upload PDFs and build it first.")
                st.stop()
                
            answer, sources = answer_question(vs, user_q)

        st.markdown("**Reply:**")
        st.write(answer)

        if sources:
            src_text = ", ".join([f"{s or 'PDF'} p.{p}" for s, p in sources])
            st.caption(f"Sources: {src_text}")


if __name__ == "__main__":
    main()