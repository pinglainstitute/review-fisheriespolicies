#!/usr/bin/env python
"""
Launch the AI Fisheries Manager Streamlit application.

Usage:
  # Test mode (default) — interactive document upload, all features visible
  python launch_streamlit.py

  # Production mode — core docs pre-loaded from a directory
  python launch_streamlit.py --mode production --docs-dir /path/to/core_pdfs
"""
import sys
import os
import argparse

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

os.chdir(os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser(description="Launch AI Fisheries Manager")
parser.add_argument("--docs-dir", type=str, default=None,
                    help="Path to a directory of core PDF documents (production mode).")
parser.add_argument("--mode", type=str, choices=["production", "test"], default="test",
                    help="Deployment mode: 'production' or 'test' (default).")
parser.add_argument("--port", type=int, default=8501,
                    help="Streamlit server port (default 8501).")
args = parser.parse_args()

# Build the streamlit CLI argv, forwarding app-specific flags after '--'
sys.argv = [
    "streamlit", "run", "llm2_updated.py",
    f"--server.port={args.port}",
    "--",  # everything after this is passed to the app script
    f"--mode={args.mode}",
]
if args.docs_dir:
    sys.argv.append(f"--docs-dir={args.docs_dir}")

from streamlit.web import cli as stcli

if __name__ == '__main__':
    sys.exit(stcli.main())


