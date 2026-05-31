import time
import torch
import numpy as np
import re
import tempfile
import streamlit as st

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

from sentence_transformers import CrossEncoder

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

st.set_page_config(
    page_title="Hybrid RAG Assistant",
    layout="wide"
)


st.title("Hybrid RAG Assistant")

st.markdown(
    "Upload a PDF and ask questions using Hybrid RAG + TinyLlama"
)


MODEL_NAME="MODEL_NAME="TinyLlama/TinyLlama-1.1B-Chat-v1.0""


@st.cache_resource
def load_model():

    tokenizer=AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    tokenizer.pad_token=tokenizer.eos_token

    quant_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=True
    )

    model=AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        quantization_config=quant_config
    )

    return tokenizer,model


tokenizer,model=load_model()


@st.cache_resource
def load_embedding():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


embedding=load_embedding()


@st.cache_resource
def load_reranker():

    return CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )


reranker=load_reranker()


uploaded_file=st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)


if uploaded_file:

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as tmp_file:

        tmp_file.write(
            uploaded_file.read()
        )

        pdf_path=tmp_file.name


    loader=PyPDFLoader(pdf_path)

    pages=loader.load()


    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )


    documents=text_splitter.split_documents(
        pages
    )


    vector_store=FAISS.from_documents(
        documents,
        embedding
    )


    chunk_texts=[
        doc.page_content
        for doc in documents
    ]


    tokenized_chunks=[
        re.findall(r'\w+',text.lower())
        for text in chunk_texts
    ]


    bm25=BM25Okapi(tokenized_chunks)


    def hybrid_search(query,k=3):

        vector_results=vector_store.similarity_search(
            query,
            k=k
        )

        vector_texts=[
            doc.page_content
            for doc in vector_results
        ]


        tokenized_query=re.findall(
            r'\w+',
            query.lower()
        )


        bm25_scores=bm25.get_scores(
            tokenized_query
        )


        top_bm25_indices=np.argsort(
            bm25_scores
        )[::-1][:k]


        bm25_texts=[
            chunk_texts[i]
            for i in top_bm25_indices
        ]


        combined_results=list(
            set(vector_texts+bm25_texts)
        )


        pairs=[
            [query,text]
            for text in combined_results
        ]


        scores=reranker.predict(pairs)


        ranked_results=sorted(
            zip(combined_results,scores),
            key=lambda x:x[1],
            reverse=True
        )


        return ranked_results[:k]


    def build_prompt(query,retrieved_docs):

        context="\n".join(
            [doc for doc,score in retrieved_docs]
        )

        prompt=f"""
<s>[INST]

Answer the question using ONLY the context below.

Context:
{context}

Question:
{query}

[/INST]
"""

        return prompt


    def generate_response(query):

        retrieval_start=time.time()

        retrieved_docs=hybrid_search(query)

        retrieval_end=time.time()


        prompt=build_prompt(
            query,
            retrieved_docs
        )


        inputs=tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(model.device)


        generation_start=time.time()

        outputs=model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )

        generation_end=time.time()


        response=tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )


        retrieval_time=(
            retrieval_end-retrieval_start
        )

        generation_time=(
            generation_end-generation_start
        )

        gpu_memory=(
            torch.cuda.memory_allocated()/1024**3
        )

        return {
            "response":response,
            "retrieval_time":retrieval_time,
            "generation_time":generation_time,
            "gpu_memory":gpu_memory,
            "sources":retrieved_docs
        }


    query=st.text_input(
        "Ask a question"
    )


    if st.button("Generate Answer"):

        if query.strip()=="":

            st.warning(
                "Please enter a question"
            )

        else:

            with st.spinner(
                "Generating response..."
            ):

                result=generate_response(
                    query
                )


            st.subheader("Answer")

            st.write(result["response"])


            col1,col2,col3=st.columns(3)

            with col1:

                st.metric(
                    "Retrieval Time",
                    f"{result['retrieval_time']:.2f}s"
                )

            with col2:

                st.metric(
                    "Generation Time",
                    f"{result['generation_time']:.2f}s"
                )

            with col3:

                st.metric(
                    "GPU Memory",
                    f"{result['gpu_memory']:.2f} GB"
                )


            st.subheader("Retrieved Chunks")


            for idx,(doc,score) in enumerate(
                result["sources"]
            ):

                with st.expander(
                    f"Chunk {idx+1} | Score: {score:.4f}"
                ):

                    st.write(doc)
