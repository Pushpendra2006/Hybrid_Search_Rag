import numpy as np
from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
from langchain.text_splitter import RecursiveCharacterTextSplitter

pdf = "C:/Users/pushp/Downloads/Thermodynamics (1).pdf"

loader = PyPDFLoader(pdf)

pages = loader.load()


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

documents = text_splitter.split_documents(pages)


embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


vector_store = FAISS.from_documents(
    documents,
    embedding
)


reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


chunk_texts = [
    doc.page_content
    for doc in documents
]


tokenized_chunks = [
    text.split()
    for text in chunk_texts
]


bm25 = BM25Okapi(tokenized_chunks)


def hybrid_search(query, k=3):

    vector_results = vector_store.similarity_search(
        query,
        k=k
    )

    vector_texts = [
        doc.page_content
        for doc in vector_results
    ]


    tokenized_query = query.split()

    bm25_scores = bm25.get_scores(tokenized_query)

    top_bm25_indices = np.argsort(
        bm25_scores
    )[::-1][:k]


    bm25_texts = [
        chunk_texts[i]
        for i in top_bm25_indices
    ]


    combined_results = list(
        set(vector_texts + bm25_texts)
    )


    pairs = [
        [query, text]
        for text in combined_results
    ]


    scores = reranker.predict(pairs)


    ranked_results = sorted(
        zip(combined_results, scores),
        key=lambda x: x[1],
        reverse=True
    )


    final_results = ranked_results[:k]

    return final_results


query = "State the laws of thermodynamics"

results = hybrid_search(query)


for text, score in results:

    print(f"Score: {score:.4f}\n")

    print(text)

    print("\n" + "="*80 + "\n")
