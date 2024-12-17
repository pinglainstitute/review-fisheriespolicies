# Neurosymbolic RAG model with reasoning for the WPCFC conservation management measures for tuna and tuna like fisheries in the WCPO 


## Setup

Obtain the PDF document of fisheries  Obtain data [CMM corpus](https://cmm.wcpfc.int/)

Install the dependencies
```
pip install -f Search_model/requirements.txt
```

## Run Application

The application is built as a streamlit application.
You can execute it as follows:

```
streamlit run Search_model/llm2_updated.py
```



## Setup Process

Step 1: Obtain data [CMM corpus](https://cmm.wcpfc.int/)

Step 2: Import to langchain (Currently it uses FAIS as vector store)

Step 3: Add to vector store

Step 4: Look at using different embeddings (langchain offers different embedding models)

Step 5: Can look at using semantic search see below (we haven't pursued this)

Step 5a: Alternatively, we could do something with symbolic AI combined with RAG, some initatives to do this in the neurosymbolic AI community. I've found two Python modules that would be useful see below.
 
### symbolic AI

[Symbolic AI Python module](https://pypi.org/project/symbolicai/)

[PyReason](https://github.com/lab-v2/pyreason) also [Read the docs](https://pyreason.readthedocs.io/en/latest/)

These can be combined with LLM 

Langchain supports different LLM engines, e.g. OpenAI, Llama, etc.  

### Semantic search

Plan is now to use MPNet with the WCPFC corpus or alternatively a RAG model, currently we have the RAG model, it needs improvement and to add reasoning. We may oersue semantic search later. 

MPNET paper here [MPNet: Masked and Permuted Pre-training for Language Understanding](https://arxiv.org/abs/2004.09297)

## Data 

WCPFC CMM corpus is here [WCPFC CMMs](https://cmm.wcpfc.int/)
