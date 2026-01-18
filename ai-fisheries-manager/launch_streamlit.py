#!/usr/bin/env python
"""Launch Streamlit application"""
import sys
import os

# Fix OpenMP library conflict issue on macOS
# Multiple libraries (faiss-cpu, sentence-transformers, numpy, etc.) may link different versions of OpenMP
# Setting this environment variable allows the program to continue running (imperfect but necessary temporary solution)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Switch to script's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Set parameters
sys.argv = ["streamlit", "run", "llm2_updated.py", "--server.port=8501"]

# Import and run streamlit CLI
from streamlit.web import cli as stcli

if __name__ == '__main__':
    sys.exit(stcli.main())

