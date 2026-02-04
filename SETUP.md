# SETUP

Documenting a complete setup from scratch with a virtual environment.

## 1. Create Virtual Environment

We encountered issues installing FAISS on versions of python higher 
than 3.10 

```
virtualenv venv310 --python=/usr/local/bin/python3.10
source venv310/bin/activate
```

## 2. Install Dependencies

```
pip3 install -r Search_model/requirements.txt
```
### SWIG

Requirements installation may fail due to the dependence of FAISS
on a library called SWIG. 

## 3. Add your Google Gemini keys




