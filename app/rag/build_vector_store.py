import pandas as pd
import faiss
import pickle

from sentence_transformers import SentenceTransformer


# =========================================
# LOAD DATASET
# =========================================

df = pd.read_csv(
    "data/enterprise_retail_dataset.csv"
)


# =========================================
# CREATE TEXT DOCUMENTS
# =========================================

documents = []


for _, row in df.iterrows():

    text = f"""

    Product: {row['product_name']}

    Category: {row['category']}

    Region: {row['region']}

    Segment: {row['segment']}

    Sales: {row['sales']}

    Profit: {row['profit']}

    Discount: {row['discount']}

    """

    documents.append(text)


# =========================================
# LOAD EMBEDDING MODEL
# =========================================

print("\nLoading embedding model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("\nEmbedding model loaded!")


# =========================================
# GENERATE EMBEDDINGS
# =========================================

print("\nGenerating embeddings...")

embeddings = model.encode(
    documents,
    show_progress_bar=True
)

print("\nEmbeddings generated successfully!")


# =========================================
# CREATE FAISS INDEX
# =========================================

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(
    dimension
)

index.add(embeddings)


# =========================================
# SAVE VECTOR DATABASE
# =========================================

faiss.write_index(

    index,

    "models/retail_vector_store.index"
)


# =========================================
# SAVE DOCUMENTS
# =========================================

with open(

    "models/retail_documents.pkl",

    "wb"
) as f:

    pickle.dump(documents, f)


# =========================================
# FINAL MESSAGE
# =========================================

print("\nVector store created successfully!")

print("\nDocuments saved successfully!")