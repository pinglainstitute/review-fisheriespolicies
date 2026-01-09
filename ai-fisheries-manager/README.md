# AI Fisheries Manager 🐟

An intelligent document Q&A system for fisheries policy management, powered by Google Gemini AI and RAG (Retrieval-Augmented Generation).

**Author:** Jiaming Yang

![Pingla Institute](https://pingla.org.au/images/Pingala_Logo_option_7.png)

## ✨ Features

- 📄 **PDF Document Processing** - Upload and process multiple fisheries policy documents
- 🔍 **Intelligent Search** - FAISS vector indexing for fast and accurate retrieval
- 🤖 **AI-Powered Q&A** - Get precise answers with source citations using Gemini 2.5 Flash
- 📊 **Advanced Retrieval** - MMR (Maximal Marginal Relevance) for diverse, non-redundant results
- 🎯 **Source Attribution** - Every answer includes references to specific pages and documents

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher
- Google/Gemini API key ([Get one here](https://makersuite.google.com/app/apikey))

### One-Command Setup (Recommended)
```bash
bash setup_llm_env.sh
```

This script will:
1. Create a virtual environment
2. Install all dependencies
3. Configure your API key
4. Launch the application

### Manual Setup

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd optimized_versio_byJiaming
```

2. **Create virtual environment**
```bash
python3 -m venv llm
source llm/bin/activate  # On Windows: llm\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure API key**
```bash
cp env.example .env
# Edit .env and add your API key
```

5. **Run the application**
```bash
./llm/bin/python launch_streamlit.py
```

The app will open in your browser at `http://localhost:8501`

## 📖 Usage

1. **Upload Documents**
   - Click "Browse files" or drag & drop PDF files
   - Click "Submit & Process" to index the documents

2. **Ask Questions**
   - Type your question in the input box
   - Get AI-powered answers with source citations

3. **View Sources**
   - See which pages and documents the answer came from
   - Citations are marked as [S1], [S2], etc.

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **AI Model**: Google Gemini 2.5 Flash
- **Vector Store**: FAISS
- **Embeddings**: Google Generative AI Embeddings (text-embedding-004)
- **PDF Processing**: pypdf
- **LangChain**: Document handling and text splitting

## 📁 Project Structure

```
.
├── llm2_updated.py          # Main application
├── launch_streamlit.py      # Launcher script
├── setup_llm_env.sh         # Auto-setup script
├── requirements.txt         # Python dependencies
├── env.example              # Environment template
├── README.md                # This file
├── PROJECT_FILES.md         # Detailed file documentation
└── .gitignore              # Git ignore rules
```

## 🔧 Configuration

### Temperature Settings
The default temperature is set to `1.1` for creative responses. Adjust in `llm2_updated.py`:
```python
"temperature": 1.1,  # 0.2-0.5 for more consistent answers
```

### Output Length
Maximum output tokens is set to `4096`. Adjust if needed:
```python
"max_output_tokens": 4096,
```

### Retrieval Parameters
- **k=10**: Number of document chunks retrieved
- **chunk_size=2500**: Document chunk size
- **chunk_overlap=300**: Overlap between chunks

## 🌊 About This Project

This is a critical applied research project addressing real-world fisheries management challenges in the **Western and Central Pacific Ocean (WCPO)** region, which accounts for approximately **25% of the global ocean area** (as of 2021).

### Why This Matters

This project tackles **real-world problems** that practitioners face **daily**. The WCPO fisheries management framework requires constant reference to complex policy documents, making this AI-powered system highly valuable for:

- Policy researchers and analysts
- Fisheries managers and administrators  
- Environmental compliance officers
- Legal advisors in maritime law

### Current Focus

The project is currently in the **comparison, evaluation, and optimization phase** of existing methodologies, focusing on:

- Benchmarking different RAG approaches
- Evaluating retrieval accuracy and answer quality
- Optimizing response time and relevance
- Refining citation and source attribution

### 🤝 Contributing

For team members and collaborators:

1. Read `PROJECT_FILES.md` for detailed file documentation
2. Never commit `.env` or API keys
3. Test locally before pushing
4. Use meaningful commit messages

## 📝 Notes

- First-time document processing may take a few minutes depending on PDF size
- The FAISS index is stored locally and regenerated when needed
- Requires active internet connection for API calls

## 🐛 Troubleshooting

### "Port 8501 already in use"
```bash
pkill -f "launch_streamlit.py"
# Then restart the app
```

### "API key is missing"
- Check your `.env` file exists
- Verify the API key is correct
- Ensure no extra spaces or quotes

### "Index build failed"
- Check internet connection
- Verify API key has sufficient quota
- Try with a smaller PDF first

## 📧 Contact

For questions, collaborations, or implementation inquiries, please contact the project team.

**Author:** Jiaming Yang

---

**Built with ❤️ for sustainable fisheries management in the Western and Central Pacific Ocean**

