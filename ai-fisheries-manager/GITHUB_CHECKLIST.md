# Development Notes

## Repository Management

### Files to Commit
```
llm2_updated.py
launch_streamlit.py
setup_llm_env.sh
requirements.txt
env.example
.gitignore
README.md
PROJECT_FILES.md
```

### Files to Exclude
```
.env                  # Contains API keys
*.json                # Cloud credentials
llm/                  # Virtual environment
faiss_index/          # Generated at runtime
__pycache__/          # Python cache
*.log                 # Log files
```

## Local Development

### Initial Setup
```bash
git clone <repository-url>
cd ai-fisheries-manager
bash setup_llm_env.sh
```

### Manual Setup
```bash
python3 -m venv llm
source llm/bin/activate  # On Windows: llm\Scripts\activate
pip install -r requirements.txt
cp env.example .env
# Add API key to .env
./llm/bin/python launch_streamlit.py
```

### Common Commands
```bash
# Start application
./llm/bin/python launch_streamlit.py

# Stop application
pkill -f "launch_streamlit.py"

# Reinstall dependencies
source llm/bin/activate
pip install -r requirements.txt --upgrade
```

## Security

Before committing:
- Verify `.env` is not staged: `git status`
- Check for credentials: `git diff --cached`
- Ensure `.gitignore` is properly configured

Never commit:
- API keys or tokens
- Service account credentials
- Environment-specific paths
- Generated indices or caches

## Collaboration

When contributing:
1. Test changes locally before committing
2. Use descriptive commit messages
3. Update documentation if changing functionality
4. Verify no credentials are included

## Troubleshooting

### Port already in use
```bash
lsof -ti:8501 | xargs kill -9
```

### Dependencies out of sync
```bash
rm -rf llm
python3 -m venv llm
source llm/bin/activate
pip install -r requirements.txt
```

### API quota exceeded
- Check Google Cloud Console quotas
- Wait for quota reset (typically daily)
- Consider upgrading API tier if needed
