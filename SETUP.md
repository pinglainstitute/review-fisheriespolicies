# SETUP

Documenting a complete setup from scratch with a virtual environment.

## 1. Create Virtual Environment

```
virtualenv venv --python=/usr/local/bin/python3.12
source venv/bin/activate
```

## 2. Install Dependencies

Note, you will need an updated version of pip to install
some dependencies.

[notice] A new release of pip is available: 24.3.1 -> 25.0.1
[notice] To update, run: pip install --upgrade pip

```
pip3 install -f Search_model/requirements.txt
```

## 3. Add your Google Gemini keys




