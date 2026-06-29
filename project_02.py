import time
import re
import tempfile
import torchvision
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

MODEL_NAME="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
device="cuda" if torch.cuda.is_available() else "cpu"

st.set_page_config(
    page_title="Hybrid RAG Assistant",
    layout="wide"
)

st.title("Hybrid RAG Research Assistant")
st.markdown("Upload a PDF and ask questions using Hybrid Retrieval + TinyLlama")

@st.cache_resource
def load_model():
    tokenizer=AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token=tokenizer.eos_token
    
    if device=="cuda":
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype=torch.float32
        
    model=AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        low_cpu_mem_usage=True
    ).to(device)
    return tokenizer, model

@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def load_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Initialize models
tokenizer, model = load_model()
embedding_model = load_embedding_model()
reranker = load_reranker()

def tokenize(text):
    return re.findall(r"\w+", text.lower())

@st.cache_resource(show_spinner=False)
def build_vector_store(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    documents = splitter.split_documents(pages)
    
    vector_store = FAISS.from_documents(documents, embedding_model)
    chunk_texts = [doc.page_content for doc in documents]
    tokenized_chunks = [tokenize(text) for text in chunk_texts]
    bm25 = BM25Okapi(tokenized_chunks)
    
    return vector_store, documents, chunk_texts, bm25

def hybrid_search(query, vector_store, chunk_texts, bm25, k=2):
    vector_results=vector_store.similarity_search(query, k=k)
    semantic_chunks=[doc.page_content for doc in vector_results]
    bm25_scores=bm25.get_scores(tokenize(query))
    top_indices=np.argsort(bm25_scores)[::-1][:k]
    keyword_chunks=[chunk_texts[i] for i in top_indices]
    candidates=list(set(semantic_chunks + keyword_chunks))
    pairs=[[query, chunk] for chunk in candidates]
    scores=reranker.predict(pairs)
    ranked=sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return ranked[:k]

def build_prompt(query,retrieved_docs):
    context="\n\n".join(doc for doc, score in retrieved_docs)
    return f"""<|system|>
Answer ONLY using the provided context. If the context does not contain the answer, say "I don't know".
Context:
{context}</s>
<|user|>
{query}</s>
<|assistant|>
"""

# ADDED: Evaluation Metrics Function
def evaluate_rag_response(query, context, answer):
    """
    Computes heuristic evaluation scores using the pre-loaded cross-encoder.
    MS-Marco cross-encoder outputs can range wildly, so we map them to a 0-100% scale safely.
    """
    if "i don't know" in answer.lower():
        return {"Faithfulness": 1.0, "Answer Relevance": 0.0}
        
    
    # Sigmoid function to normalize MS-Marco logit outputs between 0 and 1
    def normalize(score):
        return 1 / (1 + np.exp(-score))

    return {
        "Faithfulness": float(normalize(faithfulness_score)),
        "Answer Relevance": float(normalize(relevance_score))
    }

def generate_answer(query, vector_store, chunk_texts, bm25):
    retrieval_start = time.time()
    retrieved_docs = hybrid_search(query, vector_store, chunk_texts, bm25)
    retrieval_time = time.time() - retrieval_start
    context = "\n\n".join(doc for doc,score in retrieved_docs)
    prompt=build_prompt(query, retrieved_docs)
    inputs=tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
    
    generation_start = time.time()
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id
        )
    generation_time = time.time() - generation_start
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    # ADDED: Calculate metrics evaluation step
    metrics = evaluate_rag_response(query, context, response)
    return response, retrieved_docs, retrieval_time, generation_time, metrics

uploaded_file=st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:
    if "vector_store" not in st.session_state:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            pdf_path = tmp.name
            
        with st.spinner("Processing PDF..."):
            vs, docs, chunks, bm = build_vector_store(pdf_path)
            st.session_state.vector_store = vs
            st.session_state.chunk_texts = chunks
            st.session_state.bm25 = bm
            st.success("PDF processed successfully!")

    query=st.text_input("Ask a question")
    
    if st.button("Generate Answer") and query:
        if "vector_store" in st.session_state:
            with st.spinner("Thinking..."):
                answer,sources,retrieval_time,generation_time,metrics=generate_answer(
                    query,
                    st.session_state.vector_store,
                    st.session_state.chunk_texts,
                    st.session_state.bm25
                )
            
            st.subheader("Answer")
            st.write(answer)
            
            st.subheader("Evaluation Metrics")
            met1, met2, met3, met4 = st.columns(4)
            met1.metric("Faithfulness(Groundedness)",f"{metrics['Faithfulness']*100:.1f}%")
            met2.metric("Answer Relevance",f"{metrics['Answer Relevance']*100:.1f}%")
            met3.metric("Retrieval Latency",f"{retrieval_time:.2f}s")
            met4.metric("Generation Latency",f"{generation_time:.2f}s")
            
            st.subheader("Retrieved Sources")
            for idx, (doc,score) in enumerate(sources, start=1):
                with st.expander(f"Source {idx} | Reranker Score:{score:.4f}"):
                    st.write(doc)
        else: 
            st.error("Please upload a PDF first.")
