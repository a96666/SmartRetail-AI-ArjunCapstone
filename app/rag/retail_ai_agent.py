import faiss
import pickle
import numpy as np

from sentence_transformers import SentenceTransformer


# =========================================
# LOAD VECTOR DATABASE
# =========================================

print("\nLoading vector database...")

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

print("\nAI Retail Assistant Ready!")


# =========================================
# SEARCH FUNCTION
# =========================================

def search_documents(query, top_k=5):

    # Convert query into embedding
    query_embedding = model.encode(
        [query]
    )


    # Search similar vectors
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
# MAIN CHAT LOOP
# =========================================

while True:

    print("\n=================================")

    query = input(
        "\nAsk Retail AI Assistant: "
    )


    # Exit condition
    if query.lower() == "exit":

        print("\nClosing AI Assistant...")

        break


    # Search relevant docs
    retrieved_docs = search_documents(
        query
    )


    # Display AI Response
    print("\nAI Assistant Response:\n")


    for i, doc in enumerate(
        retrieved_docs,
        start=1
    ):

        print(f"\nResult {i}:")

        print(doc)

        print("-" * 40)