#!/usr/bin/env python
"""启动 Streamlit 应用"""
import sys
import os

# 修复 macOS 上的 OpenMP 库冲突问题
# 多个库（faiss-cpu, sentence-transformers, numpy 等）可能链接了不同版本的 OpenMP
# 设置此环境变量允许程序继续运行（虽然不完美，但是必要的临时解决方案）
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 切换到脚本所在目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 设置参数
sys.argv = ["streamlit", "run", "llm2_updated.py", "--server.port=8501"]

# 导入并运行 streamlit CLI
from streamlit.web import cli as stcli

if __name__ == '__main__':
    sys.exit(stcli.main())

