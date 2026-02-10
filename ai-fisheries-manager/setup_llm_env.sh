#!/usr/bin/env bash
set -euo pipefail

echo "=== AI Fisheries Manager • Environment Setup ==="

# 1) 找 Python
if command -v python3 >/dev/null 2>&1; then
  PY=python3
elif command -v python >/dev/null 2>&1; then
  PY=python
else
  echo "❌ 未找到 Python。请先安装 Python 3.9+"
  exit 1
fi
echo "• Using Python: $($PY -V)"

# 2) 创建虚拟环境 llm（如已存在则跳过）
if [ ! -d "llm" ]; then
  echo "• Creating venv: llm"
  $PY -m venv llm
else
  echo "• venv 'llm' already exists (skip create)"
fi

# 3) 激活虚拟环境
# shellcheck disable=SC1091
source llm/bin/activate
echo "• Activated venv: $(which python)"

# 4) 写/更新 requirements.txt（若已存在则保留）
REQ=requirements.txt
if [ ! -f "$REQ" ]; then
  echo "• Creating default $REQ"
  cat > "$REQ" <<'REQS'
streamlit>=1.30.0,<1.40.0
python-dotenv>=1.0.1
protobuf<6

# —— LangChain / Gemini（彼此兼容的一组版本）——
langchain==0.2.14
langchain-community==0.2.12
langchain-google-genai==1.0.10
google-generativeai==0.7.2

# —— 向量库 / PDF / 安全 IO —— 
faiss-cpu>=1.7.4
pypdf>=4.2.0
pyarrow>=14.0.1
REQS
else
  echo "• Found existing $REQ (keeping as-is)"
fi

# 5) 安装依赖
echo "• Upgrading pip"
pip install -U pip >/dev/null
echo "• Installing requirements"
pip install --no-cache-dir -U -r "$REQ"

# 6) 写入 .env（如缺则创建；如已有则只在缺 key 时补齐两种变量）
ENV_FILE=".env"
if [ -f "$ENV_FILE" ]; then
  echo "• Existing .env detected"
  if ! grep -q '^GOOGLE_API_KEY=' "$ENV_FILE" && ! grep -q '^GEMINI_API_KEY=' "$ENV_FILE"; then
    read -rp "请输入你的 GOOGLE/GEMINI API KEY: " GKEY
    {
      echo "GOOGLE_API_KEY=$GKEY"
      echo "GEMINI_API_KEY=$GKEY"
    } >> "$ENV_FILE"
    echo "• GOOGLE_API_KEY + GEMINI_API_KEY appended to .env"
  fi
else
  read -rp "请输入你的 GOOGLE/GEMINI API KEY: " GKEY
  {
    echo "GOOGLE_API_KEY=$GKEY"
    echo "GEMINI_API_KEY=$GKEY"
  } > "$ENV_FILE"
  echo "• .env created"
fi

echo "• Running quick checks"

# ---------- 可靠加载 .env 并桥接两种变量 ----------
set +u
# zsh 有时需要 .env 末尾换行；若无则补一个
if [ -f ".env" ]; then
  tail -c1 .env | read -r _ || printf "\n" >> .env
  # 将 .env 导出为环境变量
  set -a; . ./.env; set +a
fi

# 互补：只设了其中一个时，自动补齐另一个（安全展开，不触发 set -u）
: "${GOOGLE_API_KEY:=${GEMINI_API_KEY:-}}"
: "${GEMINI_API_KEY:=${GOOGLE_API_KEY:-}}"
export GOOGLE_API_KEY GEMINI_API_KEY
set -u

echo "• Keys: GOOGLE_API_KEY=$([ -n "${GOOGLE_API_KEY:-}" ] && echo set || echo missing), GEMINI_API_KEY=$([ -n "${GEMINI_API_KEY:-}" ] && echo set || echo missing)"

# ---------- Python 自检 ----------
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

# 8) 启动应用（如果文件存在）
APP="llm2_updated.py"
if [ -f "$APP" ]; then
  echo "• Launching Streamlit app"
  exec streamlit run "$APP"
else
  echo "⚠️ 未找到 $APP，请确认脚本与该文件在同一目录。"
  echo "你也可以稍后手动运行："
  echo "  source llm/bin/activate && streamlit run llm2_updated.py"
fi