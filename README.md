# Hybrid RAG Research Assistant 

[![Live Demo](https://img.shields.io/badge/Live-Demo-brightgreen?style=for-the-badge&logo=streamlit)](https://huggingface.co/spaces/PUSHPENDRA2006/HYBRID_RAG_AGENT)
[![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python)](https://www.python.org)
[![Framework](https://img.shields.io/badge/LangChain-⚡-orange?style=for-the-badge)](https://github.com/langchain-ai/langchain)

An enterprise-grade, deployment-ready **Hybrid Retrieval-Augmented Generation (RAG)** system designed to ingest complex multi-document PDFs and deliver context-aware, verifiable answers with ultra-low latency. 
By combining dense semantic search, sparse keyword matching, and cross-encoder reranking, this assistant eliminates hallucinations and provides precision-engineered document intelligence.


## Key Features & Architecture

 **Hybrid Retrieval Engine:** Combines **FAISS** (Dense/Semantic) and **BM25** (Sparse/Keyword) to capture both conceptual meaning and exact term matches.
 **Cross-Encoder Reranking:** Integrates a reranking pipeline to evaluate and re-order the top retrieved contexts, maximizing relevance before feeding data to the LLM.
 **Optimized Ingestion Pipeline:** High-performance PDF parser featuring **recursive character chunking** to preserve contextual boundaries.
 **Source-Aware Tracking:** Every response is explicitly mapped back to its source document and page snippet for 100% auditability.
 **Lightweight LLM Orchestration:** Minimized system inference latency through streamlined prompt engineering and optimized token management.



##  Tech Stack

 **Core Framework:** LangChain
 **Vector Database:** FAISS
 **Keyword Search:** BM25
 **Frontend:** Streamlit
 **Embeddings & LLMs:** Hugging Face Transformers / OpenAI API (Adaptable)
 **Language:** Python


##  System Workflow & Performance
Multi-PDF Ingestion]
│
▼
[Recursive Chunking] ──► [Embedding Gen] ──► [FAISS (Dense Search)] ──┐
│                                                               │
└───────────────────────────────────► [BM25 (Sparse Search)] ──┼─► [CrossEncoder Rerank] ──► [Lightweight LLM] ──► [Audited Output]
