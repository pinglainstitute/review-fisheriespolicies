# Project Structure

## Core Files

### Application
- `llm2_updated.py` - Main application implementing the RAG-based Q&A system
- `launch_streamlit.py` - Application launcher
- `setup_llm_env.sh` - Environment setup script (creates venv, installs dependencies, configures API key)

### Configuration
- `requirements.txt` - Python dependencies
- `env.example` - Environment variable template (copy to `.env` and add your API key)
- `.gitignore` - Git ignore rules

### Documentation
- `README.md` - Project overview and setup instructions

## Dependencies

```
streamlit>=1.30.0,<1.40.0
python-dotenv>=1.0.1
langchain==0.2.14
langchain-community==0.2.12
langchain-google-genai==1.0.10
google-generativeai==0.7.2
faiss-cpu>=1.7.4
pypdf>=4.2.0
pyarrow>=14.0.1
```

## Excluded from Repository

The following are generated at runtime or contain sensitive data:
- `llm/` - Virtual environment (recreate with `python3 -m venv llm`)
- `faiss_index/` - Vector index (generated on first document upload)
- `.env` - API keys (never commit this file)
- `__pycache__/`, `*.pyc` - Python cache files

## Setup

1. Clone the repository
2. Run `bash setup_llm_env.sh`
3. Add your Google/Gemini API key when prompted

Or manually:
```bash
python3 -m venv llm
source llm/bin/activate
pip install -r requirements.txt
cp env.example .env
# Edit .env with your API key
./llm/bin/python launch_streamlit.py
```

## Runtime Files

After first run, the following will be created:
- `faiss_index/` - Vector database for document embeddings
- `streamlit.log` - Application logs (if configured)

These are excluded from version control and will be regenerated as needed.
