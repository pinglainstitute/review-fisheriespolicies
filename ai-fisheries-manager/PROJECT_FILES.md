# AI Fisheries Manager - Project Files

## 📦 **Core Application Files** (必须上传)

### 主程序
- **`llm2_updated.py`** - 主应用程序（当前使用的版本）
  - Streamlit UI
  - PDF 文档处理
  - FAISS 向量索引
  - Gemini AI 集成
  - RAG 问答系统

### 启动脚本
- **`launch_streamlit.py`** - Python 启动脚本（推荐使用）
  - 直接调用 Streamlit CLI
  - 跨平台兼容

- **`setup_llm_env.sh`** - 自动环境配置脚本
  - 创建虚拟环境
  - 安装依赖
  - 配置 API key
  - 自动启动应用

### 配置文件
- **`requirements.txt`** - Python 依赖包列表
  ```
  streamlit>=1.30.0,<1.40.0
  python-dotenv>=1.0.1
  protobuf<6
  langchain==0.2.14
  langchain-community==0.2.12
  langchain-google-genai==1.0.10
  google-generativeai==0.7.2
  faiss-cpu>=1.7.4
  pypdf>=4.2.0
  pyarrow>=14.0.1
  ```

- **`.env`** - 环境变量配置（⚠️ 不要上传到 GitHub！）
  - GOOGLE_API_KEY
  - GEMINI_API_KEY

---

## 🗂️ **辅助脚本文件** (可选上传)

- `start_app.sh` - 简单启动脚本
- `run_app.sh` - 另一个启动脚本
- `test_streamlit.py` - 测试文件

---

## 📚 **文档文件** (建议上传)

### PDF 测试文档
- `compiled-conservation-measures-and-resolutions.pdf` (39.3MB)
- `Fisheries Management Documents/` - 其他渔业政策文档
  - 可以选择性上传一些示例文档

---

## 🚫 **不应上传到 GitHub 的文件/文件夹**

### 虚拟环境
- `llm/` - Python 虚拟环境（自动生成）
- `LLMenv/` - 旧虚拟环境
- `pyreason-env/` - 其他环境

### 索引和缓存
- `faiss_index/` - FAISS 向量索引（运行时生成）
- `__pycache__/` - Python 缓存
- `*.pyc` - 编译的 Python 文件
- `streamlit.log` - 日志文件

### 敏感文件
- `.env` - API 密钥（⚠️ 绝对不能上传！）
- `*.json` - Google Cloud 凭证文件
  - `ComputeEngine.json`
  - `enduring-lane-443604-p5-*.json`

### 旧版本文件
- `llm2_updated_pre.py` - 旧版本
- `llm3updated.py` ~ `llm7updated.py` - 开发过程中的旧版本
- `llmupdated.py`, `llmupdatedLast.py` - 更早的版本
- `Original/` - 原始备份
- `llmforpdf/` - 旧版本目录

---

## 📋 **推荐的 GitHub 项目结构**

```
ai-fisheries-manager/
├── README.md                          # 项目说明
├── requirements.txt                   # Python 依赖
├── .env.example                       # 环境变量模板
├── .gitignore                         # Git 忽略规则
├── llm2_updated.py                    # 主应用
├── launch_streamlit.py                # 启动脚本
├── setup_llm_env.sh                   # 环境配置脚本
├── docs/                              # 文档目录
│   ├── INSTALLATION.md               # 安装指南
│   └── USER_GUIDE.md                 # 使用指南
└── sample_data/                       # 示例数据（可选）
    └── sample.pdf                    # 小的示例 PDF
```

---

## 🔧 **团队成员需要做什么**

### 1. 克隆项目后
```bash
git clone <your-repo-url>
cd ai-fisheries-manager
```

### 2. 配置环境（自动方式）
```bash
bash setup_llm_env.sh
# 脚本会提示输入 API key
```

### 3. 手动配置（如果需要）
```bash
# 创建虚拟环境
python3 -m venv llm

# 激活虚拟环境
source llm/bin/activate  # macOS/Linux
# 或
llm\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt

# 配置 API key
cp .env.example .env
# 编辑 .env 文件，添加你的 API key
```

### 4. 运行应用
```bash
./llm/bin/python launch_streamlit.py
# 或
streamlit run llm2_updated.py
```

---

## 📝 **重要说明**

1. **API Key 管理**
   - 每个团队成员需要自己的 Google/Gemini API key
   - 绝对不要把 API key 提交到 GitHub
   - 使用 `.env.example` 作为模板

2. **文档大小**
   - 大型 PDF 文件（>50MB）建议不上传到 GitHub
   - 可以使用 Git LFS 或单独提供下载链接

3. **虚拟环境**
   - 每个成员在本地创建自己的虚拟环境
   - 不要上传 `llm/` 目录

4. **索引文件**
   - `faiss_index/` 会在首次上传文档时自动生成
   - 不需要提交到 Git

---

## 🎯 **快速开始命令**

```bash
# 一行命令完成所有配置并启动
bash setup_llm_env.sh
```

