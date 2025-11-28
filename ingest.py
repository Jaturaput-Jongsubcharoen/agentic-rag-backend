# ingest.py
import os
import pandas as pd
from dotenv import load_dotenv

# LOCAL embeddings – no Google quota
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

load_dotenv()



# ------------------------------------------------------
# 1. Load CSV files (train, val, test)
# ------------------------------------------------------
def load_csv_files():
    files = [
        os.path.join("data", "train_split.csv"),
        os.path.join("data", "val_split.csv"),
        os.path.join("data", "test_split.csv")
    ]

    dfs = []
    for fp in files:
        if os.path.exists(fp):
            df = pd.read_csv(fp)
            df["source"] = fp
            dfs.append(df)
        else:
            print(f"Missing file: {fp}")

    return dfs


# ------------------------------------------------------
# 2. Convert rows → Document objects
# ------------------------------------------------------
def dataframe_to_documents(df):
    docs = []

    for _, row in df.iterrows():

        metadata = {
            "filepath": row.get("filepath", "unknown"),
            "label": row.get("label", "unknown"),
            "label_idx": row.get("label_idx", "unknown"),
            "source": row.get("source", "unknown"),
        }

        text = (
            f"Image Path: {metadata['filepath']}\n"
            f"Label: {metadata['label']}\n"
            f"Label Index: {metadata['label_idx']}\n"
            f"Source CSV: {metadata['source']}\n"
        )

        docs.append(Document(page_content=text, metadata=metadata))

    return docs


# ------------------------------------------------------
# 3. Chunk text (RAG-ready)
# ------------------------------------------------------
def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
    )
    return splitter.split_documents(docs)


# ------------------------------------------------------
# 4. Create FAISS vectorstore using HuggingFace embeddings
# ------------------------------------------------------
def build_vectorstore(chunks):

    print("Initializing HuggingFace Embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("Creating FAISS vectorstore...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    os.makedirs("backend/vectorstore", exist_ok=True)
    vectorstore.save_local("backend/vectorstore")

    print("Vectorstore saved successfully!")


# ------------------------------------------------------
# MAIN
# ------------------------------------------------------
if __name__ == "__main__":
    print("🚀 Starting ingestion...")

    dfs = load_csv_files()
    all_docs = []

    for df in dfs:
        all_docs.extend(dataframe_to_documents(df))

    chunks = split_docs(all_docs)
    build_vectorstore(chunks)

    print("Ingestion complete! Ready for RAG.")