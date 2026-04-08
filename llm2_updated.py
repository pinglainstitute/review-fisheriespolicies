# llm2_updated.py
# llm2_updated.py
import os
import shutil
import time
import json
from datetime import datetime
import streamlit as st
import tiktoken
from typing import List, Tuple, Dict, Optional

from dotenv import load_dotenv
load_dotenv()

# ---------- API Key Configuration ----------
# Support Streamlit Cloud secrets, .env file, and environment variables
API_KEY = (
    st.secrets.get("GOOGLE_API_KEY", None) if hasattr(st, "secrets") else None
) or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not API_KEY:
    st.error(
        "API key is missing. Set GOOGLE_API_KEY in:\n"
        "- Streamlit Cloud Secrets (for hosted deployment), or\n"
        "- .env file / environment variable (for local development)"
    )
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

# (If you want to use LangChain's conversation chain, keep these two; current example uses direct client calls)
# from langchain.chains.question_answering import load_qa_chain
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.prompts import PromptTemplate

INDEX_DIR = "faiss_index"
METADATA_FILE = os.path.join(INDEX_DIR, "index_metadata.json")
FEEDBACK_FILE = "feedback_data.json"
PARAMS_CONFIG_FILE = "system_params.json"
CREDENTIALS_FILE = "credentials.json"
EMBED_MODEL = "models/text-embedding-004"

# ---------- CLI Configuration ----------
import argparse

def parse_app_args():
    """Parse command-line arguments passed after '--' in the streamlit run command."""
    parser = argparse.ArgumentParser(description="AI Fisheries Manager")
    parser.add_argument(
        "--docs-dir",
        type=str,
        default=None,
        help="Path to a directory of core PDF documents to load on startup (production mode).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["production", "test"],
        default="test",
        help="Deployment mode: 'production' (hardwired docs, restricted user UI) or 'test' (full access).",
    )
    # Streamlit passes extra args; use parse_known_args to ignore them
    args, _ = parser.parse_known_args()
    return args

APP_ARGS = parse_app_args()

# ---------- Simple Authentication / Login ----------

def _load_user_credentials() -> Dict[str, Dict[str, str]]:
    """
    Load user credentials with the following priority:
      1. credentials.json file in the working directory
      2. APP_USERS environment variable (JSON string)
      3. Individual APP_USERNAME / APP_PASSWORD / APP_ROLE env vars
    """
    users: Dict[str, Dict[str, str]] = {}

    # Priority 1a: Streamlit Cloud secrets (APP_USERS key)
    try:
        secrets_users = st.secrets.get("APP_USERS", None)
        if secrets_users:
            parsed = json.loads(secrets_users) if isinstance(secrets_users, str) else dict(secrets_users)
            for username, data in parsed.items():
                data = dict(data) if not isinstance(data, dict) else data
                if "password" in data:
                    users[username] = {
                        "password": str(data["password"]),
                        "role": str(data.get("role", "user")),
                    }
    except Exception:
        pass

    if users:
        return users

    # Priority 1b: credentials.json file
    if os.path.exists(CREDENTIALS_FILE):
        try:
            with open(CREDENTIALS_FILE, "r", encoding="utf-8") as f:
                creds = json.load(f)
            parsed = creds.get("users", creds)
            for username, data in parsed.items():
                if isinstance(data, dict) and "password" in data:
                    users[username] = {
                        "password": str(data["password"]),
                        "role": str(data.get("role", "user")),
                    }
        except Exception:
            pass

    if users:
        return users

    # Priority 2: APP_USERS env var
    users_env = os.getenv("APP_USERS")
    if users_env:
        try:
            parsed = json.loads(users_env)
            for username, data in parsed.items():
                if isinstance(data, dict) and "password" in data:
                    users[username] = {
                        "password": str(data.get("password", "")),
                        "role": str(data.get("role", "user")),
                    }
        except Exception:
            pass

    if users:
        return users

    # Priority 3: single-user fallback
    username = os.getenv("APP_USERNAME", "admin")
    password = os.getenv("APP_PASSWORD", "secret")
    role = os.getenv("APP_ROLE", "admin")
    users[username] = {"password": password, "role": role}

    return users


def authenticate_user(username: str, password: str) -> Optional[Dict[str, str]]:
    """Return user record (including role) if credentials are valid, else None."""
    users = _load_user_credentials()
    record = users.get(username)
    if not record:
        return None
    if password != record.get("password"):
        return None
    return {"username": username, "role": record.get("role", "user")}


def require_login():
    """
    Simple login gate in front of the main app.

    - Shows a dedicated login screen when user is not authenticated
    - On success, stores {"is_authenticated": True, "username": ..., "role": ...}
      in st.session_state["auth"] and re-runs the app so the main UI appears.
    - Includes a small note about this being a demo-grade auth layer.
    """
    if "auth" not in st.session_state:
        st.session_state["auth"] = {
            "is_authenticated": False,
            "username": None,
            "role": None,
        }

    auth_state = st.session_state["auth"]
    if auth_state.get("is_authenticated"):
        return  # Already logged in

    st.title("AI Fisheries Manager – Login")
    st.write(
        "This demo is protected by a simple username/password login. "
        "Credentials are configured via environment variables."
    )

    username = st.text_input("Username", key="login_username")
    password = st.text_input("Password", type="password", key="login_password")

    col_login, col_info = st.columns([1, 2])
    with col_login:
        if st.button("Log in", type="primary", key="login_button"):
            user = authenticate_user(username.strip(), password)
            if user:
                st.session_state["auth"] = {
                    "is_authenticated": True,
                    "username": user["username"],
                    "role": user["role"],
                }
                st.success(f"Welcome, {user['username']}!")
                st.rerun()
            else:
                st.error("Invalid username or password.")
    with col_info:
        st.caption(
            "For production use, replace this with a stronger auth mechanism "
            "(e.g. SSO / OAuth2, reverse-proxy auth, or Streamlit's enterprise features)."
        )

    # Stop rendering the rest of the app while not authenticated
    st.stop()

# ---------- Token counting ----------
# Initialize tokenizer (using cl100k_base which is close to Gemini's tokenization)
_tokenizer = None

def get_tokenizer():
    """Get or initialize tokenizer for token counting."""
    global _tokenizer
    if _tokenizer is None:
        # Use cl100k_base (GPT-4 tokenizer) as approximation for Gemini
        # This is close enough for truncation purposes
        _tokenizer = tiktoken.get_encoding("cl100k_base")
    return _tokenizer

def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken."""
    try:
        encoding = get_tokenizer()
        return len(encoding.encode(text))
    except Exception:
        # Fallback: rough estimate (1 token ≈ 4 characters for English)
        return len(text) // 4

# ---------- Query Enhancement ----------
def expand_query(query: str) -> List[str]:
    """
    Expand query with synonyms and variations for better retrieval.
    Uses simple rule-based expansion for fisheries domain terms.
    """
    # Common fisheries terminology variations
    expansions = {
        'vessel': ['ship', 'boat', 'craft'],
        'fishing': ['fishery', 'harvesting', 'catch'],
        'limit': ['restriction', 'quota', 'ceiling', 'maximum'],
        'conservation': ['protection', 'preservation', 'management'],
        'measure': ['regulation', 'rule', 'provision', 'requirement'],
        'longline': ['long line', 'long-line'],
        'fresh fish': ['fresh catch', 'freshly caught fish'],
    }

    expanded_queries = [query]  # Original query first

    # Add variations
    query_lower = query.lower()
    for term, synonyms in expansions.items():
        if term in query_lower:
            for synonym in synonyms:
                expanded = query_lower.replace(term, synonym)
                if expanded != query_lower:
                    expanded_queries.append(expanded)

    # Also try with common connectors
    if len(expanded_queries) == 1:
        # If no expansions found, try adding common terms
        expanded_queries.append(f"{query} policy")
        expanded_queries.append(f"{query} regulation")

    return expanded_queries[:3]  # Limit to 3 variations

def rerank_documents(query: str, docs: List, top_k: int = 10) -> List:
    """
    Re-rank retrieved documents using cross-encoder for better precision.
    Falls back to original order if cross-encoder unavailable.
    """
    if not docs:
        return []

    try:
        from sentence_transformers import CrossEncoder

        # Initialize cross-encoder (lightweight model) - lazy loading
        if not hasattr(rerank_documents, 'model'):
            try:
                rerank_documents.model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            except Exception as e:
                # If model download fails, disable re-ranking for this session
                rerank_documents.model = None
                return docs[:top_k]

        # If model failed to load, skip re-ranking
        if rerank_documents.model is None:
            return docs[:top_k]

        # Limit document content length to avoid memory issues
        max_content_len = 1000
        pairs = []
        for doc in docs:
            content = doc.page_content[:max_content_len] if len(doc.page_content) > max_content_len else doc.page_content
            pairs.append([query, content])

        # Get scores
        try:
            scores = rerank_documents.model.predict(pairs)
        except Exception as e:
            # If prediction fails, return original order
            return docs[:top_k]

        # Sort by score (descending)
        scored_docs = list(zip(docs, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # Return top-k re-ranked documents
        return [doc for doc, score in scored_docs[:top_k]]

    except ImportError:
        # Fallback: return original order if sentence-transformers not available
        return docs[:top_k]
    except Exception as e:
        # Silent fallback - don't show warning to avoid UI clutter
        return docs[:top_k]

# ---------- Feedback Collection and Parameter Optimization ----------
def save_feedback(question: str, answer: str, sources: List[Tuple], feedback_value: int, feedback_type: str = "rating"):
    """Save user feedback to a JSON file

    Args:
        question: User's question
        answer: System's answer
        sources: List of source documents
        feedback_value: Feedback value (1-5 rating, 1=👍, 0=👎)
        feedback_type: Type of feedback ("rating" or "thumbs")
    """
    try:
        feedback_data = []
        if os.path.exists(FEEDBACK_FILE):
            try:
                with open(FEEDBACK_FILE, 'r', encoding='utf-8') as f:
                    feedback_data = json.load(f)
            except:
                feedback_data = []

        entry = {
            "question": question,
            "answer": answer[:500],  # Limit answer length
            "sources": sources,
            "feedback_value": feedback_value,
            "feedback_type": feedback_type,
            "timestamp": datetime.now().isoformat()
        }

        feedback_data.append(entry)

        # Save to file
        with open(FEEDBACK_FILE, 'w', encoding='utf-8') as f:
            json.dump(feedback_data, f, indent=2, ensure_ascii=False)

        return True
    except Exception as e:
        st.warning(f"Failed to save feedback: {str(e)}")
        return False

def load_feedback_data():
    """Load feedback data from file"""
    try:
        if os.path.exists(FEEDBACK_FILE):
            with open(FEEDBACK_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        st.warning(f"Failed to load feedback data: {str(e)}")
    return []

def analyze_feedback():
    """Analyze feedback data and return statistics and optimization suggestions"""
    feedback_data = load_feedback_data()
    if not feedback_data:
        return None

    # Calculate average rating
    ratings = [f['feedback_value'] for f in feedback_data if f.get('feedback_type') == 'rating']
    thumbs_up = sum(1 for f in feedback_data if f.get('feedback_type') == 'thumbs' and f.get('feedback_value') == 1)
    thumbs_down = sum(1 for f in feedback_data if f.get('feedback_type') == 'thumbs' and f.get('feedback_value') == 0)

    stats = {
        "total_feedback": len(feedback_data),
        "average_rating": sum(ratings) / len(ratings) if ratings else None,
        "thumbs_up": thumbs_up,
        "thumbs_down": thumbs_down,
        "positive_rate": (thumbs_up / (thumbs_up + thumbs_down)) if (thumbs_up + thumbs_down) > 0 else None
    }

    return stats

def get_optimized_params():
    """Get optimized system parameters based on feedback data"""
    feedback_data = load_feedback_data()
    if not feedback_data:
        # Return default parameters
        return {
            "retrieval_k": 10,
            "mmr_lambda": 0.2,
            "temperature": 1.1,
            "max_context_tokens": 6000
        }

    # Analyze feedback to identify patterns
    positive_count = 0
    negative_count = 0

    for entry in feedback_data:
        if entry.get('feedback_type') == 'rating':
            if entry.get('feedback_value', 0) >= 4:
                positive_count += 1
            elif entry.get('feedback_value', 0) <= 2:
                negative_count += 1
        elif entry.get('feedback_type') == 'thumbs':
            if entry.get('feedback_value') == 1:
                positive_count += 1
            else:
                negative_count += 1

    # Adjust parameters based on feedback rate
    total_feedback = len(feedback_data)
    if total_feedback < 5:
        # Not enough feedback data, use default parameters
        return {
            "retrieval_k": 10,
            "mmr_lambda": 0.2,
            "temperature": 1.1,
            "max_context_tokens": 6000
        }

    positive_rate = positive_count / total_feedback

    # If positive feedback rate is low (<50%), increase retrieval count and diversity
    if positive_rate < 0.5:
        return {
            "retrieval_k": 12,  # Increase number of documents to retrieve
            "mmr_lambda": 0.15,  # Lower lambda to increase diversity
            "temperature": 1.0,  # Slightly lower temperature to improve accuracy
            "max_context_tokens": 6000
        }
    # If positive feedback rate is high (>70%), keep or optimize current parameters
    elif positive_rate > 0.7:
        return {
            "retrieval_k": 10,
            "mmr_lambda": 0.2,
            "temperature": 1.1,
            "max_context_tokens": 6000
        }
    # For medium feedback rate, use balanced parameters
    else:
        return {
            "retrieval_k": 10,
            "mmr_lambda": 0.2,
            "temperature": 1.0,
            "max_context_tokens": 6000
        }

def save_system_params(params: Dict):
    """Save system parameters to configuration file"""
    try:
        with open(PARAMS_CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(params, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        st.warning(f"Failed to save system params: {str(e)}")
        return False

def load_system_params():
    """Load system parameters (from config file or optimized based on feedback)

    Strategy:
    1. If config file exists and feedback data has less than 5 entries, use saved parameters
    2. If feedback data has >= 5 entries, re-optimize parameters based on feedback
    3. If no config file exists, use default or optimized parameters
    """
    try:
        feedback_data = load_feedback_data()
        feedback_count = len(feedback_data) if feedback_data else 0

        # If there's enough feedback data (>=5 entries), optimize parameters based on feedback
        if feedback_count >= 5:
            optimized_params = get_optimized_params()
            # Save optimized parameters
            save_system_params(optimized_params)
            return optimized_params

        # If config file exists but feedback is insufficient, use saved parameters
        if os.path.exists(PARAMS_CONFIG_FILE):
            with open(PARAMS_CONFIG_FILE, 'r', encoding='utf-8') as f:
                saved_params = json.load(f)
                return saved_params
    except Exception as e:
        st.warning(f"Failed to load system params: {str(e)}")

    # Return default parameters or optimized parameters
    default_params = get_optimized_params()
    save_system_params(default_params)  # Save default parameters
    return default_params

# ---------- Conversation Context Management ----------
CONVERSATION_HISTORY_LIMIT = 5  # Keep the most recent 5 conversation turns

def init_conversation_history():
    """Initialize conversation history"""
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

def add_to_history(question: str, answer: str, sources: List[Tuple], timing_info: Dict):
    """Add question-answer pair to conversation history"""
    if 'conversation_history' not in st.session_state:
        init_conversation_history()

    entry = {
        "question": question,
        "answer": answer,
        "sources": sources,
        "timing_info": timing_info,
        "timestamp": datetime.now().isoformat()
    }

    st.session_state.conversation_history.append(entry)

    # Limit history size, only keep the most recent N turns
    if len(st.session_state.conversation_history) > CONVERSATION_HISTORY_LIMIT:
        st.session_state.conversation_history = st.session_state.conversation_history[-CONVERSATION_HISTORY_LIMIT:]

def get_recent_context(max_turns: int = 3) -> str:
    """Get recent conversation context to enhance query"""
    if 'conversation_history' not in st.session_state:
        return ""

    history = st.session_state.conversation_history[-max_turns:]
    if not history:
        return ""

    context_parts = []
    for entry in history:
        context_parts.append(f"Q: {entry['question']}\nA: {entry['answer']}")

    return "\n\n".join(context_parts)

def build_enhanced_query(user_question: str, include_history: bool = True) -> str:
    """Build enhanced query that includes historical context

    Strategy:
    1. If question contains pronouns or reference words (like "this policy", "above"), extract key info from history
    2. Otherwise, mainly use current question for retrieval
    3. Historical context is mainly used in prompt for answer generation, not for retrieval
    """
    if not include_history:
        return user_question

    # Check if question contains pronouns or reference words
    reference_words = ['this', 'that', 'these', 'those', 'it', 'they', 'above', 'previous', 'earlier', 'mentioned', 'same']
    question_lower = user_question.lower()
    has_reference = any(word in question_lower for word in reference_words)

    if has_reference and 'conversation_history' in st.session_state and len(st.session_state.conversation_history) > 0:
        # Extract key entities and topic words from recent conversation
        recent_entry = st.session_state.conversation_history[-1]

        # Extract keywords from recent question (remove stop words)
        recent_question = recent_entry['question']
        # Extract nouns and important words (simple approach: words longer than 4 chars or important terms)
        recent_words = [w for w in recent_question.split() if len(w) > 4 or w.lower() in ['cmm', 'cmmo', 'fao', 'un', 'tuna', 'fish']]

        # If recent question has relevant keywords, use them to enhance query
        if recent_words:
            context_keywords = ' '.join(recent_words[:3])  # Maximum 3 keywords
            enhanced_query = f"{context_keywords} {user_question}"
            return enhanced_query.strip()

    # By default, use original question
    return user_question

# ---------- Source Formatting ----------
def format_sources(sources):
    """
    Format sources display, use concise format if all from same file.

    Examples:
    Single file: compiled-conservation-measures-and-resolutions.pdf (p.41, p.38, p.52)
    Multiple files: doc1.pdf (p.1, p.2), doc2.pdf (p.5, p.6)
    """
    if not sources:
        return ""

    # Group by file name
    file_pages = {}
    for source, page in sources:
        file_name = source or "PDF"
        if file_name not in file_pages:
            file_pages[file_name] = []
        file_pages[file_name].append(page)

    # Format page numbers for each file
    formatted_parts = []
    for file_name, pages in file_pages.items():
        # Sort page numbers
        pages = sorted(set(pages), key=lambda x: int(str(x).replace('?', '0')))
        # Format as p.1, p.2, p.3
        page_str = ", ".join([f"p.{p}" for p in pages])
        formatted_parts.append(f"{file_name} ({page_str})")

    return ", ".join(formatted_parts)

# ---------- Document Metadata Management ----------
def save_document_metadata(pdf_files, page_count, chunk_count, processed_time=None):
    """Save document metadata to JSON file"""
    try:
        # Make sure index directory exists
        os.makedirs(INDEX_DIR, exist_ok=True)

        # Build metadata
        metadata = {
            "processed_time": processed_time or datetime.now().isoformat(),
            "page_count": page_count,
            "chunk_count": chunk_count,
            "documents": []
        }

        # Add information for each document
        for pdf_file in pdf_files:
            file_name = getattr(pdf_file, "name", "uploaded.pdf")
            file_size = getattr(pdf_file, "size", 0)
            metadata["documents"].append({
                "name": file_name,
                "size_mb": round(file_size / (1024 * 1024), 2) if file_size > 0 else 0
            })

        # Save to JSON file
        with open(METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        return True
    except Exception as e:
        st.warning(f"Failed to save document metadata: {str(e)}")
        return False

def load_document_metadata():
    """Load document metadata from JSON file"""
    try:
        if os.path.exists(METADATA_FILE):
            with open(METADATA_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        st.warning(f"Failed to load document metadata: {str(e)}")
    return None

# ---------- Helper Functions ----------
def get_docs_with_meta(pdf_files):
    """Read multiple PDFs, extract page by page as Documents, preserve source and page numbers; with text cleaning."""
    docs = []
    for f in pdf_files:
        reader = PdfReader(f)
        name = getattr(f, "name", "uploaded.pdf")

        for i, page in enumerate(reader.pages):
            raw_txt = page.extract_text() or ""
            txt = " ".join(raw_txt.split())

            if len(txt) < 40:
                continue

            docs.append(
                Document(
                    page_content=txt,
                    metadata={"source": name, "page": i + 1}
                )
            )
    return docs


def get_docs_from_directory(dir_path: str) -> List[Document]:
    """Load all PDFs from a local directory and return Documents with metadata."""
    import glob as glob_mod
    docs = []
    pdf_paths = sorted(glob_mod.glob(os.path.join(dir_path, "*.pdf")))
    if not pdf_paths:
        return docs
    for pdf_path in pdf_paths:
        try:
            reader = PdfReader(pdf_path)
            name = os.path.basename(pdf_path)
            for i, page in enumerate(reader.pages):
                raw_txt = page.extract_text() or ""
                txt = " ".join(raw_txt.split())
                if len(txt) < 40:
                    continue
                docs.append(
                    Document(
                        page_content=txt,
                        metadata={"source": name, "page": i + 1},
                    )
                )
        except Exception as e:
            st.warning(f"Failed to read {os.path.basename(pdf_path)}: {e}")
    return docs


def load_core_documents_if_needed():
    """
    If --docs-dir is set, load core PDFs from that directory and build the
    FAISS index (unless one already exists). Returns True if core docs were
    loaded/already present, False otherwise.
    """
    docs_dir = APP_ARGS.docs_dir
    if not docs_dir:
        return False

    if not os.path.isdir(docs_dir):
        st.error(f"Documents directory not found: {docs_dir}")
        return False

    index_exists = os.path.isdir(INDEX_DIR) and os.path.exists(
        os.path.join(INDEX_DIR, "index.faiss")
    )
    if index_exists:
        return True

    st.info(f"Loading core documents from `{docs_dir}` ...")
    raw_docs = get_docs_from_directory(docs_dir)
    if not raw_docs:
        st.warning("No PDF pages extracted from the documents directory.")
        return False

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1800,
        chunk_overlap=250,
        length_function=len,
    )
    chunks = splitter.split_documents(raw_docs)
    st.info(f"Extracted {len(raw_docs)} pages → {len(chunks)} chunks. Building index...")

    if os.path.isdir(INDEX_DIR):
        shutil.rmtree(INDEX_DIR, ignore_errors=True)
    build_or_load_vector_store.clear()
    vs = build_or_load_vector_store(chunks)
    if vs is None:
        st.error("Failed to build index from core documents.")
        return False

    st.success(f"Core document index ready ({len(chunks)} chunks).")
    return True

# def answer_question(vs, user_question: str):
#     """Similarity search + Gemini 2.5 Flash (optional Thinking)"""
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
#             thinking_config=types.ThinkingConfig(thinking_budget=30)  # Lower to speed up/save cost
#         ),
#     )
#     answer = (resp.text or "").strip() if hasattr(resp, "text") else ""
#     if not answer:
#         answer = "No output produced."

#     # Return answer and sources for UI display
#     sources = [(d.metadata.get("source"), d.metadata.get("page")) for d in docs]
#     return answer, sources
def robust_retrieve(vs, query, k=10, mmr_lambda=0.2):
    """
    Enhanced retrieval with query expansion, MMR diversity, and re-ranking.

    Steps:
    1. Query expansion for better recall
    2. Initial retrieval with similarity scoring
    3. MMR for diversity
    4. Cross-encoder re-ranking for precision
    """
    try:
        # Step 1: Query expansion
        expanded_queries = expand_query(query)

        # Step 2: Retrieve candidates from all query variations
        all_candidates = []
        seen_content = set()

        for expanded_query in expanded_queries:
            # Get initial candidates with scores
            pairs = vs.similarity_search_with_score(expanded_query, k=20)
            for doc, score in pairs:
                # Deduplicate by content
                content_hash = hash(doc.page_content[:100])
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    all_candidates.append((doc, score))

        # Sort all candidates by score (lower score = more similar)
        all_candidates.sort(key=lambda x: x[1])

        # Step 3: Apply MMR for diversity (use top candidates as fetch_k)
        fetch_k = min(30, len(all_candidates))
        if fetch_k > 0:
            # Get top candidates for MMR
            top_candidates = [doc for doc, _ in all_candidates[:fetch_k]]

            # Apply MMR on top candidates
            try:
                mmr_docs = vs.max_marginal_relevance_search(
                    query, k=k, fetch_k=fetch_k, lambda_mult=mmr_lambda
                )
            except Exception:
                # If MMR fails, use top candidates directly
                mmr_docs = top_candidates[:k]
        else:
            mmr_docs = []

        # Step 4: Re-rank with cross-encoder for better precision
        if mmr_docs:
            try:
                final_docs = rerank_documents(query, mmr_docs, top_k=k)
            except Exception:
                # If re-ranking fails, use MMR results
                final_docs = mmr_docs[:k]
        else:
            # Fallback: use top candidates if MMR returned empty
            final_docs = [doc for doc, _ in all_candidates[:k]] if all_candidates else []

        return final_docs if final_docs else []

    except Exception as e:
        # Fallback: simple similarity search
        try:
            return vs.similarity_search(query, k=k)
        except Exception:
            # Last resort: return empty list
            return []

def answer_question(vs, user_question: str, conversation_history: List = None):

    """More robust retrieval + structured LLM answer generation + citation markup

    Args:
        vs: Vector store
        user_question: Current user question
        conversation_history: List of previous conversation entries (optional)
    """

    # Record start time
    start_time = time.time()
    retrieval_start = time.time()

    # Load optimized system parameters
    system_params = load_system_params()
    retrieval_k = system_params.get("retrieval_k", 10)
    mmr_lambda = system_params.get("mmr_lambda", 0.2)
    max_context_tokens = system_params.get("max_context_tokens", 6000)
    temperature = system_params.get("temperature", 1.1)

    # Build enhanced query (includes historical context)
    enhanced_query = build_enhanced_query(user_question, include_history=True)

    # Use enhanced query and optimized parameters for retrieval
    docs = robust_retrieve(vs, enhanced_query, k=retrieval_k, mmr_lambda=mmr_lambda)
    retrieval_time = time.time() - retrieval_start

    # Build context and add source information
    blocks, used = [], []

    if not docs:
        st.warning("No documents retrieved for this query.")
        # If no documents retrieved, return directly
        generation_start = time.time()
        generation_time = time.time() - generation_start
        total_time = time.time() - start_time
        return "I don't know.", [], {
            "retrieval_time": retrieval_time,
            "generation_time": generation_time,
            "total_time": total_time
        }
    # Token-based truncation: Gemini 2.5 Flash has ~8192 token context window
    # Reserve ~2000 tokens for prompt template and response, use ~6000 for context
    # Use optimized parameters
    total_tokens = 0
    truncated_count = 0

    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "PDF")
        pg  = d.metadata.get("page", "?")
        block = f"[S{i} | {src} p.{pg}]\n{d.page_content.strip()}"
        block_tokens = count_tokens(block)

        # Check if adding this block would exceed token limit
        if total_tokens + block_tokens > max_context_tokens:
            truncated_count = len(docs) - i
            break

        blocks.append(block)
        used.append((src, pg))
        total_tokens += block_tokens

    # Warn user if truncation occurred
    if truncated_count > 0:
        st.warning(f"Context truncated: {truncated_count} document chunk(s) omitted to fit token limit ({total_tokens}/{max_context_tokens} tokens used).")

    context_text = "\n\n".join(blocks)

    # Build conversation history section (for generation)
    # Note: Need to control history context length to avoid exceeding token limit
    conversation_context = ""
    if conversation_history and len(conversation_history) > 0:
        recent_history = conversation_history[-2:]  # Most recent 2 turns
        history_parts = []
        history_tokens = 0
        MAX_HISTORY_TOKENS = 500  # Reserve maximum 500 tokens for conversation history

        for entry in reversed(recent_history):  # From most recent to oldest
            q_text = entry['question']
            a_text = entry['answer'][:300]  # Limit answer length to avoid being too long

            history_entry = f"Previous Q: {q_text}\nPrevious A: {a_text[:300]}..."
            entry_tokens = count_tokens(history_entry)

            # If adding this history entry would exceed limit, stop adding
            if history_tokens + entry_tokens > MAX_HISTORY_TOKENS:
                break

            history_parts.insert(0, history_entry)  # Maintain chronological order
            history_tokens += entry_tokens

        if history_parts:
            conversation_context = "\n\nPrevious conversation:\n" + "\n\n".join(history_parts) + "\n"

    # Build prompt (includes conversation history)
    # If there's conversation history, adjust prompt to leverage context
    history_instruction = ""
    if conversation_context:
        history_instruction = "Consider the previous conversation context when answering. If the current question refers to something mentioned earlier (using words like 'this', 'that', 'above', 'previous'), use the conversation history to understand what is being referred to.\n\n"

    prompt = f"""
You are an expert fisheries policy assistant.
Use ONLY the context below. If the context contains relevant facts, answer concisely and cite sources like [S1], [S2] at the end of sentences derived from them.
Only say "I don't know." if the context truly does not contain the answer.
{history_instruction}{conversation_context}
Context:
{context_text}

Question:
{user_question}

Answer (with citations):
""".strip()

    # Record start time for answer generation
    generation_start = time.time()

    # Call Gemini model to generate answer (using optimized temperature)
    model = genai.GenerativeModel("gemini-2.5-flash")
    resp = model.generate_content(
        prompt,
        generation_config={
            "temperature": temperature,  # Use optimized temperature
            "max_output_tokens": 4096,  # Increase output length limit to avoid answer truncation
        },
    )
    # Extract answer text
    answer = (getattr(resp, "text", "") or "").strip()
    if not answer:
        answer = "I don't know."

    # Calculate generation time
    generation_time = time.time() - generation_start
    total_time = time.time() - start_time

    # If answer not found, show most relevant sections
    if answer.strip() == "I don't know." and used:
        hints = [f"• {s or 'PDF'} p.{p}" for s,p in used[:3]]
        st.info("Closest sections:\n" + "\n".join(hints))

    return answer, used, {
        "retrieval_time": retrieval_time,
        "generation_time": generation_time,
        "total_time": total_time,
        "system_params": {  # Return used parameters for logging
            "retrieval_k": retrieval_k,
            "mmr_lambda": mmr_lambda,
            "temperature": temperature,
            "max_context_tokens": max_context_tokens
        }
    }

@st.cache_resource(show_spinner=False)
def build_or_load_vector_store(_documents=None):
    """
    Load existing index if available; otherwise build when documents are provided.
    Uses batch processing to avoid memory issues with large document sets.
    Returns None if no index exists and no documents are provided (no exception thrown).
    """
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)
    except Exception as e:
        st.error(f"Embeddings initialization failed: {str(e)}")
        return None

    # Try to load existing index
    if os.path.isdir(INDEX_DIR):
        try:
            return FAISS.load_local(
                INDEX_DIR, embeddings, allow_dangerous_deserialization=False
            )
        except Exception as e:
            # Version or format incompatible, continue to rebuild
            st.warning(f"Failed to load existing index, will rebuild: {str(e)}")
            pass

    # No local index; build if documents provided, otherwise return None
    if _documents:
        try:
            total_chunks = len(_documents)
            # Note: Don't display UI elements inside cached function, display before calling function instead

            # Reduce batch size and increase intermediate save frequency to avoid timeouts and API rate limits
            BATCH_SIZE = 20  # Reduce batch size to lower API call pressure
            SAVE_INTERVAL = 3  # Save every 3 batches to avoid losing progress
            API_DELAY = 1.0  # Delay between API calls (in seconds) to avoid rate limits

            vs = None
            processed_count = 0

            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            time_text = st.empty()

            start_time = time.time()

            for batch_start in range(0, total_chunks, BATCH_SIZE):
                batch_end = min(batch_start + BATCH_SIZE, total_chunks)
                batch_docs = _documents[batch_start:batch_end]

                # Update progress
                progress = (batch_end / total_chunks)
                progress_bar.progress(progress)
                elapsed_time = time.time() - start_time
                estimated_total = elapsed_time / progress if progress > 0 else 0
                remaining_time = estimated_total - elapsed_time

                status_text.text(f"Processing chunks {batch_start + 1}-{batch_end} of {total_chunks}...")
                time_text.text(f"Elapsed: {int(elapsed_time)}s | Estimated remaining: {int(remaining_time)}s")

                # API call retry mechanism
                max_retries = 3
                retry_count = 0
                batch_success = False

                while retry_count < max_retries and not batch_success:
                    try:
                        # Add delay to avoid API rate limits (except first batch)
                        if batch_start > 0:
                            time.sleep(API_DELAY)

                        # Process current batch
                        if vs is None:
                            # First batch: create new index
                            vs = FAISS.from_documents(batch_docs, embedding=embeddings)
                        else:
                            # Subsequent batches: add to existing index
                            batch_index = FAISS.from_documents(batch_docs, embedding=embeddings)
                            vs.merge_from(batch_index)
                            # Clear batch index to free memory
                            del batch_index

                        processed_count += len(batch_docs)
                        batch_success = True

                        # Periodically save intermediate results to avoid losing progress if connection drops
                        batch_number = (batch_start // BATCH_SIZE) + 1
                        if batch_number % SAVE_INTERVAL == 0 or batch_end >= total_chunks:
                            try:
                                # Save to temporary directory
                                temp_index_dir = f"{INDEX_DIR}_temp"
                                vs.save_local(temp_index_dir)
                                # If save successful, move to main directory
                                if os.path.exists(os.path.join(temp_index_dir, "index.faiss")):
                                    if os.path.isdir(INDEX_DIR):
                                        shutil.rmtree(INDEX_DIR, ignore_errors=True)
                                    shutil.move(temp_index_dir, INDEX_DIR)
                                    status_text.text(f"Saved checkpoint: {processed_count}/{total_chunks} chunks processed")
                            except Exception as save_err:
                                # Intermediate save failure doesn't affect main process, continue processing
                                pass

                    except Exception as batch_error:
                        retry_count += 1
                        if retry_count < max_retries:
                            wait_time = retry_count * 2  # Exponential backoff
                            status_text.text(f"Batch {batch_start + 1}-{batch_end} failed (attempt {retry_count}/{max_retries}). Retrying in {wait_time}s...")
                            time.sleep(wait_time)
                        else:
                            st.warning(f"Failed to process batch {batch_start + 1}-{batch_end} after {max_retries} attempts: {str(batch_error)}")
                            # Continue processing next batch, don't interrupt entire process
                            break

                # Force refresh Streamlit interface to keep connection alive
                if batch_start % (BATCH_SIZE * 2) == 0:
                    status_text.empty()
                    status_text.text(f"Processing chunks {batch_start + 1}-{batch_end} of {total_chunks}...")

            # Complete progress bar
            progress_bar.progress(1.0)
            time_text.empty()
            status_text.text("Saving final index to disk...")

            # Final save of index
            if vs is not None and processed_count > 0:
                try:
                    vs.save_local(INDEX_DIR)
                    # Verify index file was created successfully
                    index_file = os.path.join(INDEX_DIR, "index.faiss")
                    if os.path.exists(index_file):
                        status_text.text(f"Index saved successfully! ({processed_count}/{total_chunks} chunks)")
                        status_text.empty()
                        progress_bar.empty()
                        return vs
                    else:
                        st.error("Index file was not created properly.")
                        return None
                except Exception as save_error:
                    st.error(f"Failed to save index: {str(save_error)}")
                    return None
            else:
                st.error(f"Failed to build index: Only {processed_count}/{total_chunks} chunks were processed.")
                return None

        except Exception as e:
            st.error(f"Index building failed: {str(e)}")
            import traceback
            st.error(f"Detailed error:\n```\n{traceback.format_exc()}\n```")
            return None

    # No index and no documents - let upper level handle friendly prompt
    return None

# ---------- Streamlit UI ----------

def _is_admin() -> bool:
    """Check if the current authenticated user has the admin role."""
    return st.session_state.get("auth", {}).get("role") == "admin"


def _show_feedback_ui() -> bool:
    """Determine whether to show feedback UI to the current user."""
    if APP_ARGS.mode == "production":
        return _is_admin()
    return True  # In test mode everyone sees feedback


def _render_sidebar_admin(auth_state: dict):
    """Render sidebar sections visible only to admins."""

    st.header("Documents")

    # --- Upload supplementary documents ---
    upload_label = (
        "Upload supplementary PDFs"
        if APP_ARGS.docs_dir
        else "Upload PDF files then click 'Submit & Process'"
    )
    pdf_docs = st.file_uploader(upload_label, type=["pdf"], accept_multiple_files=True)

    if st.button("Submit & Process", type="primary", help="Extract, chunk, and index PDFs"):
        if not pdf_docs:
            st.error("Please upload at least one PDF.")
        else:
            with st.spinner("Extracting & indexing..."):
                try:
                    st.info(f"Extracting {len(pdf_docs)} PDF file(s)...")
                    raw_docs = get_docs_with_meta(pdf_docs)
                    if not raw_docs:
                        st.error("Failed to extract text content from PDFs.")
                    else:
                        st.success(f"Successfully extracted {len(raw_docs)} page(s)")

                        splitter = RecursiveCharacterTextSplitter(
                            chunk_size=1800,
                            chunk_overlap=250,
                            length_function=len,
                        )
                        chunks = splitter.split_documents(raw_docs)
                        st.info(f"Documents split into {len(chunks)} chunk(s)")

                        if os.path.isdir(INDEX_DIR):
                            shutil.rmtree(INDEX_DIR, ignore_errors=True)
                        build_or_load_vector_store.clear()

                        vs = build_or_load_vector_store(chunks)
                        if vs is None:
                            st.error("Index building failed. Please check error messages and retry.")
                        else:
                            st.success("Index built successfully!")
                            save_document_metadata(pdf_docs, len(raw_docs), len(chunks))
                            st.info("Index is ready! Ask questions in the main area.")
                except Exception as e:
                    st.error(f"Processing error: {str(e)}")
                    import traceback
                    st.error(f"Detailed error:\n```\n{traceback.format_exc()}\n```")

    st.divider()

    # --- Feedback statistics (admin only) ---
    feedback_stats = analyze_feedback()
    if feedback_stats and feedback_stats.get("total_feedback", 0) > 0:
        st.caption("**Feedback Statistics:**")
        if feedback_stats.get("positive_rate") is not None:
            rate = feedback_stats["positive_rate"]
            st.caption(
                f"Positive rate: {rate:.1%} "
                f"({feedback_stats.get('thumbs_up', 0)} 👍 / "
                f"{feedback_stats.get('thumbs_down', 0)} 👎)"
            )
        if feedback_stats.get("average_rating") is not None:
            st.caption(f"Average rating: {feedback_stats['average_rating']:.2f}/5")
        st.caption(f"Total feedback: {feedback_stats['total_feedback']}")
        st.divider()

    # --- System parameters (admin only) ---
    params = load_system_params()
    with st.expander("System Parameters", expanded=False):
        st.json(params)

    st.divider()

    # --- Index status / document metadata ---
    _render_index_status()


def _render_index_status():
    """Show index and document metadata in the sidebar."""
    index_exists = os.path.isdir(INDEX_DIR) and os.path.exists(
        os.path.join(INDEX_DIR, "index.faiss")
    )
    if index_exists:
        metadata = load_document_metadata()
        if metadata:
            st.caption("**Indexed documents:**")
            doc_info = [
                f"• {doc['name']} ({doc.get('size_mb', 0)} MB)"
                for doc in metadata.get("documents", [])
            ]
            if doc_info:
                for info in doc_info[:3]:
                    st.caption(info)
                if len(doc_info) > 3:
                    st.caption(f"... and {len(doc_info) - 3} more document(s)")
                st.caption(
                    f"{metadata.get('page_count', 0)} page(s) | "
                    f"{metadata.get('chunk_count', 0)} chunk(s)"
                )
            else:
                st.caption("FAISS index detected")
        else:
            st.caption("FAISS index detected")
    else:
        st.caption("No index yet.")


def main():
    st.set_page_config(page_title="AI Fisheries Manager", page_icon=None)

    # ---- Authentication gate ----
    require_login()

    auth_state = st.session_state.get("auth", {})
    user_role = auth_state.get("role", "user")
    is_admin = _is_admin()

    st.title("AI Fisheries Manager")
    st.image("https://pingla.org.au/images/Pingala_Logo_option_7.png", width=300)

    if APP_ARGS.mode == "production" and APP_ARGS.docs_dir:
        st.caption("Production mode — core documents pre-loaded")
    elif APP_ARGS.mode == "test":
        st.caption("Test / development mode")

    # ---- Load core documents from --docs-dir if provided ----
    load_core_documents_if_needed()

    # ---- Sidebar ----
    with st.sidebar:
        # User info + logout
        if auth_state.get("is_authenticated"):
            st.markdown(f"**Logged in as:** `{auth_state.get('username', '')}`")
            st.caption(f"Role: {user_role}")
            if st.button("Log out", key="logout_button"):
                st.session_state["auth"] = {
                    "is_authenticated": False,
                    "username": None,
                    "role": None,
                }
                st.rerun()

        st.divider()

        if is_admin:
            _render_sidebar_admin(auth_state)
        else:
            # Regular user sidebar: only show index status
            if APP_ARGS.docs_dir:
                st.caption("Core documents are pre-loaded by the administrator.")
            else:
                st.caption("Please ask the administrator to upload documents.")
            _render_index_status()

    # ---- Main area: Q&A ----
    index_exists = os.path.isdir(INDEX_DIR) and os.path.exists(
        os.path.join(INDEX_DIR, "index.faiss")
    )

    init_conversation_history()

    if 'user_question' not in st.session_state:
        st.session_state.user_question = ""

    if 'clear_input' in st.session_state and st.session_state.clear_input:
        if 'question_input' in st.session_state:
            del st.session_state.question_input
        st.session_state.user_question = ""
        st.session_state.clear_input = False

    user_q = st.text_input(
        "Ask the fisheries manager a question",
        key="question_input",
        placeholder="Enter your question here...",
    )

    if user_q != st.session_state.user_question:
        st.session_state.user_question = user_q

    col1, col2, col3 = st.columns([5, 1, 1])
    with col2:
        if st.button("Clear", key="clear_button"):
            st.session_state.clear_input = True
            st.rerun()
    with col3:
        if st.button("Clear History", key="clear_history_button"):
            if 'conversation_history' in st.session_state:
                st.session_state.conversation_history = []
            st.rerun()

    # Conversation history (all users can see their own)
    if 'conversation_history' in st.session_state and len(st.session_state.conversation_history) > 0:
        with st.expander(
            f"Conversation History ({len(st.session_state.conversation_history)} turns)",
            expanded=False,
        ):
            for i, entry in enumerate(st.session_state.conversation_history, 1):
                st.markdown(f"**Turn {i}:**")
                st.markdown(f"**Q:** {entry['question']}")
                st.markdown(f"**A:** {entry['answer']}")
                if entry.get('sources'):
                    src_text = format_sources(entry['sources'])
                    st.caption(f"Sources: {src_text}")
                st.divider()

    if user_q and user_q.strip():
        if not index_exists:
            if is_admin:
                st.warning("No index found. Upload PDFs and click 'Submit & Process'.")
            else:
                st.warning("No documents indexed yet. Please contact the administrator.")
            return

        with st.spinner("Retrieving & answering..."):
            vs = build_or_load_vector_store()
            if vs is None:
                build_or_load_vector_store.clear()
                vs = build_or_load_vector_store()
                if vs is None:
                    st.error("Failed to load index.")
                    st.stop()

            conversation_history = st.session_state.get('conversation_history', [])
            answer, sources, timing_info = answer_question(vs, user_q, conversation_history)
            add_to_history(user_q, answer, sources, timing_info)

        st.markdown("**Reply:**")
        st.write(answer)

        # Timing info (admin always sees; regular users see in test mode)
        if is_admin or APP_ARGS.mode == "test":
            st.caption(
                f"**Time:** Retrieval: {timing_info['retrieval_time']:.2f}s | "
                f"Generation: {timing_info['generation_time']:.2f}s | "
                f"**Total: {timing_info['total_time']:.2f}s**"
            )

        if sources:
            src_text = format_sources(sources)
            st.caption(f"**Sources:** {src_text}")

        # Feedback UI (controlled by role and mode)
        if _show_feedback_ui():
            st.markdown("---")
            st.markdown("**Was this answer helpful?**")

            current_turn = len(st.session_state.get('conversation_history', []))
            feedback_saved_key = f"feedback_saved_{current_turn}"

            if feedback_saved_key not in st.session_state:
                st.session_state[feedback_saved_key] = False

            fb_col1, fb_col2, fb_col3 = st.columns([1, 1, 6])

            if not st.session_state[feedback_saved_key]:
                with fb_col1:
                    if st.button("👍", key=f"thumbs_up_{current_turn}"):
                        save_feedback(user_q, answer, sources, 1, "thumbs")
                        st.session_state[feedback_saved_key] = True
                        get_optimized_params()
                        st.success("Thank you for your feedback!")
                        st.rerun()
                with fb_col2:
                    if st.button("👎", key=f"thumbs_down_{current_turn}"):
                        save_feedback(user_q, answer, sources, 0, "thumbs")
                        st.session_state[feedback_saved_key] = True
                        get_optimized_params()
                        st.info("Thank you for your feedback. We'll use this to improve.")
                        st.rerun()
            else:
                st.caption("Feedback submitted. Thank you!")


if __name__ == "__main__":
    main()