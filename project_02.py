import time
import re
import tempfile

import numpy as np
import streamlit as st
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

from sentence_transformers import CrossEncoder

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rank_bm25 import BM25Okapi

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

device = "cuda" if torch.cuda.is_available() else "cpu"

st.set_page_config(
page_title="Hybrid RAG Assistant",
layout="wide"
)

st.title(" Hybrid RAG Research Assistant")

st.markdown(
"Upload a PDF and ask questions using Hybrid Retrieval + TinyLlama"
)

@st.cache_resource
def load_model():

```
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME
)

return tokenizer, model
```

@st.cache_resource
def load_embedding_model():

```
return HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

@st.cache_resource
def load_reranker():

```
return CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)
```

tokenizer, model = load_model()

embedding_model = load_embedding_model()

reranker = load_reranker()

def tokenize(text):

```
return re.findall(
    r"\w+",
    text.lower()
)
```

@st.cache_resource(show_spinner=False)
def build_vector_store(pdf_path):

```
loader = PyPDFLoader(pdf_path)

pages = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

documents = splitter.split_documents(
    pages
)

vector_store = FAISS.from_documents(
    documents,
    embedding_model
)

chunk_texts = [
    doc.page_content
    for doc in documents
]

tokenized_chunks = [
    tokenize(text)
    for text in chunk_texts
]

bm25 = BM25Okapi(
    tokenized_chunks
)

return (
    vector_store,
    documents,
    chunk_texts,
    bm25
)
```

def hybrid_search(
query,
vector_store,
chunk_texts,
bm25,
k=3
):

```
vector_results = vector_store.similarity_search(
    query,
    k=k
)

semantic_chunks = [
    doc.page_content
    for doc in vector_results
]

bm25_scores = bm25.get_scores(
    tokenize(query)
)

top_indices = np.argsort(
    bm25_scores
)[::-1][:k]

keyword_chunks = [
    chunk_texts[i]
    for i in top_indices
]

candidates = list(
    set(
        semantic_chunks +
        keyword_chunks
    )
)

pairs = [
    [query, chunk]
    for chunk in candidates
]

scores = reranker.predict(
    pairs
)

ranked = sorted(
    zip(candidates, scores),
    key=lambda x: x[1],
    reverse=True
)

return ranked[:k]
```

def build_prompt(
query,
retrieved_docs
):

```
context = "\n\n".join(
    doc
    for doc, score in retrieved_docs
)

return f"""
```

Answer ONLY using the provided context.

Context:
{context}

Question:
{query}

Answer:
"""

def generate_answer(
query,
vector_store,
chunk_texts,
bm25
):

```
retrieval_start = time.time()

retrieved_docs = hybrid_search(
    query,
    vector_store,
    chunk_texts,
    bm25
)

retrieval_time = (
    time.time() -
    retrieval_start
)

prompt = build_prompt(
    query,
    retrieved_docs
)

inputs = tokenizer(
    prompt,
    return_tensors="pt",
    truncation=True,
    max_length=2048
).to(device)

generation_start = time.time()

outputs = model.generate(
    **inputs,
    max_new_tokens=200,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
    pad_token_id=tokenizer.pad_token_id
)

generation_time = (
    time.time() -
    generation_start
)

response = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
)

return (
    response,
    retrieved_docs,
    retrieval_time,
    generation_time
)
```

uploaded_file = st.file_uploader(
"Upload PDF",
type=["pdf"]
)

if uploaded_file:

```
with tempfile.NamedTemporaryFile(
    delete=False,
    suffix=".pdf"
) as tmp:

    tmp.write(
        uploaded_file.read()
    )

    pdf_path = tmp.name

with st.spinner(
    "Processing PDF..."
):

    (
        vector_store,
        documents,
        chunk_texts,
        bm25
    ) = build_vector_store(
        pdf_path
    )

query = st.text_input(
    "Ask a question"
)

if st.button(
    "Generate Answer"
) and query:

    with st.spinner(
        "Thinking..."
    ):

        (
            answer,
            sources,
            retrieval_time,
            generation_time
        ) = generate_answer(
            query,
            vector_store,
            chunk_texts,
            bm25
        )

    st.subheader("Answer")

    st.write(answer)

    col1, col2 = st.columns(2)

    col1.metric(
        "Retrieval Time",
        f"{retrieval_time:.2f}s"
    )

    col2.metric(
        "Generation Time",
        f"{generation_time:.2f}s"
    )

    st.subheader(
        "Retrieved Sources"
    )

    for idx, (
        doc,
        score
    ) in enumerate(
        sources,
        start=1
    ):

        with st.expander(
            f"Source {idx} | Score {score:.4f}"
        ):

            st.write(doc)
```
