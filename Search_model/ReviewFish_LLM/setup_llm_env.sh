#!/usr/bin/env bash
set -euo pipefail

echo "=== AI Fisheries Manager • Environment Setup ==="

# 1) Find Python
if command -v python3 >/dev/null 2>&1; then
  PY=python3
elif command -v python >/dev/null 2>&1; then
  PY=python
else
  echo "❌ Python not found. Please install Python 3.9+ first."
  exit 1
fi
echo "• Using Python: $($PY -V)"

# 2) Create a virtual environment named 'llm' (skip if it already exists)
if [ ! -d "llm" ]; then
  echo "• Creating venv: llm"
  $PY -m venv llm
else
  echo "• venv 'llm' already exists (skip create)"
fi

# 3) Activate the virtual environment
# shellcheck disable=SC1091
source llm/bin/activate
echo "• Activated venv: $(which python)"

# 4) Create or reuse requirements.txt
REQ=requirements.txt
if [ ! -f "$REQ" ]; then
  echo "• Creating default $REQ"
  cat > "$REQ" <<'REQS'
streamlit>=1.30.0,<1.40.0
python-dotenv>=1.0.1
protobuf<6

# --- LangChain / Gemini (a compatible version set) ---
langchain==0.2.14
langchain-community==0.2.12
langchain-google-genai==1.0.10
google-generativeai==0.7.2

# --- Vector store / PDF / Safe IO ---
faiss-cpu>=1.7.4
pypdf>=4.2.0
pyarrow>=14.0.1
REQS
else
  echo "• Found existing $REQ (keeping as-is)"
fi

# 5) Install dependencies
echo "• Upgrading pip"
pip install -U pip >/dev/null
echo "• Installing requirements"
pip install --no-cache-dir -U -r "$REQ"

# 6) Create or update .env with API keys
ENV_FILE=".env"
if [ -f "$ENV_FILE" ]; then
  echo "• Existing .env detected"
  if ! grep -q '^GOOGLE_API_KEY=' "$ENV_FILE" && ! grep -q '^GEMINI_API_KEY=' "$ENV_FILE"; then
    read -rp "Please enter your GOOGLE/GEMINI API KEY: " GKEY
    {
      echo "GOOGLE_API_KEY=$GKEY"
      echo "GEMINI_API_KEY=$GKEY"
    } >> "$ENV_FILE"
    echo "• GOOGLE_API_KEY + GEMINI_API_KEY appended to .env"
  fi
else
  read -rp "Please enter your GOOGLE/GEMINI API KEY: " GKEY
  {
    echo "GOOGLE_API_KEY=$GKEY"
    echo "GEMINI_API_KEY=$GKEY"
  } > "$ENV_FILE"
  echo "• .env created"
fi

echo "• Running quick checks"

# ---------- Reliable .env loading & key bridging ----------
set +u
# Ensure the .env file ends with a newline (important for zsh)
if [ -f ".env" ]; then
  tail -c1 .env | read -r _ || printf "\n" >> .env
  # Export .env variables
  set -a; . ./.env; set +a
fi

# Bridge GOOGLE_API_KEY and GEMINI_API_KEY if only one is set
: "${GOOGLE_API_KEY:=${GEMINI_API_KEY:-}}"
: "${GEMINI_API_KEY:=${GOOGLE_API_KEY:-}}"
export GOOGLE_API_KEY GEMINI_API_KEY
set -u

echo "• Keys: GOOGLE_API_KEY=$([ -n "${GOOGLE_API_KEY:-}" ] && echo set || echo missing), GEMINI_API_KEY=$([ -n "${GEMINI_API_KEY:-}" ] && echo set || echo missing)"

# ---------- Python self-check ----------
python - <<'PY'
import os, sys
ok = True

gkey = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
print(" - GOOGLE_API_KEY:", "set" if os.getenv("GOOGLE_API_KEY") else "missing")
print(" - GEMINI_API_KEY :", "set" if os.getenv("GEMINI_API_KEY") else "missing")

try:
    import pypdf
    print(" - pypdf:", pypdf.__version__)
except Exception as e:
    ok = False
    print(" - pypdf import error:", e)

try:
    import google.generativeai as genai
    if not gkey:
        raise RuntimeError("No API key found in env")
    genai.configure(api_key=gkey)
    models = list(genai.list_models())
    print(f" - GenerativeAI models: {len(models)}")
except Exception as e:
    ok = False
    print(" - GenerativeAI check failed:", e)

try:
    import langchain, langchain_community, langchain_google_genai
    print(f" - langchain: {langchain.__version__}")
except Exception as e:
    ok = False
    print(" - LangChain import failed:", e)

if not ok:
    sys.exit(2)
PY

# 8) Launch the app (if the file exists)
APP="llm2_updated.py"
if [ -f "$APP" ]; then
  echo "• Launching Streamlit app"
  exec streamlit run "$APP"
else
  echo "⚠️ $APP not found. Please make sure this script is in the same directory."
  echo "You can also run manually later with:"
  echo "  source llm/bin/activate && streamlit run llm2_updated.py"
fi