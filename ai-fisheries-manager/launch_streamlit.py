#!/usr/bin/env python
"""Launch Streamlit application"""
import sys
import os
import json
import time

# region agent log
_DBG_LOG_PATH = "/Users/onlytash/Downloads/review-fisheriespolicies-main/.cursor/debug.log"
def _dbg(hypothesisId: str, location: str, message: str, data: dict):
    try:
        with open(_DBG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "sessionId": "debug-session",
                "runId": "pre-fix",
                "hypothesisId": hypothesisId,
                "location": location,
                "message": message,
                "data": data,
                "timestamp": int(time.time() * 1000),
            }, ensure_ascii=False) + "\n")
    except Exception:
        pass
# endregion agent log

# Fix OpenMP library conflict issue on macOS
# Multiple libraries (faiss-cpu, sentence-transformers, numpy, etc.) may link different versions of OpenMP
# Setting this environment variable allows the program to continue running (imperfect but necessary temporary solution)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Switch to script's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Set parameters
_dbg("H1", "launch_streamlit.py:pre-argv", "entry", {
    "python": sys.executable,
    "cwd": os.getcwd(),
    "orig_argv": sys.argv[:],
})
sys.argv = ["streamlit", "run", "llm2_updated.py", "--server.port=8501"]
_dbg("H1", "launch_streamlit.py:post-argv", "argv_overwritten", {
    "new_argv": sys.argv[:],
})

# Import and run streamlit CLI
from streamlit.web import cli as stcli

if __name__ == '__main__':
    _dbg("H1", "launch_streamlit.py:main", "calling_stcli_main", {"note": "about_to_start_streamlit_cli"})
    sys.exit(stcli.main())


