#!/usr/bin/env python
"""启动 Streamlit 应用"""
import sys
import os

# 切换到脚本所在目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 设置参数
sys.argv = ["streamlit", "run", "llm2_updated.py", "--server.port=8501"]

# 导入并运行 streamlit CLI
from streamlit.web import cli as stcli

if __name__ == '__main__':
    sys.exit(stcli.main())

