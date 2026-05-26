import faiss
import pickle
import numpy as np

from sentence_transformers import SentenceTransformer


# =========================================
# LOAD VECTOR DATABASE
# =========================================

index = faiss.read_index(
    "models/retail_vector_store.index"
)


# =========================================
# LOAD DOCUMENTS
# =========================================

with open(
    "models/retail_documents.pkl",
    "rb"
) as f:

    documents = pickle.load(f)


# =========================================
# LOAD EMBEDDING MODEL
# =========================================

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# =========================================
# SEARCH FUNCTION
# =========================================

def search_documents(query, top_k=5):

    query_embedding = model.encode(
        [query]
    )

    distances, indices = index.search(
        np.array(query_embedding),
        top_k
    )

    results = []

    for idx in indices[0]:

        results.append(
            documents[idx]
        )

    return results


# =========================================
# RAG RESPONSE FUNCTION
# =========================================

def retail_rag_response(query):

    retrieved_docs = search_documents(query)

    context = "\n".join(retrieved_docs)

    response = f"""
Based on retrieved retail knowledge:

{context}

Answer:
The query suggests insights related to retail analytics,
customer behavior, inventory trends,
and business operations.
"""

    return response