# rag_graph.py
# ------------------------------------------------------
# COMP258 – Lab Assignment #4
# Agentic RAG Backend (NO GEMINI, NO API CALLS)
# ------------------------------------------------------

import os
from dotenv import load_dotenv

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

# ------------------------------------------------------
# Load FAISS vectorstore
# ------------------------------------------------------
def get_retriever():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectorstore = FAISS.load_local(
        "backend/vectorstore",
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vectorstore.as_retriever(search_kwargs={"k": 3})


retriever = get_retriever()


# ------------------------------------------------------
# Simple RAG answer generator (no Gemini API)
# ------------------------------------------------------
def rag_answer(question):

    # 1. Retrieve relevant documents
    docs = retriever.get_relevant_documents(question)

    context_text = "\n\n".join([d.page_content for d in docs])

    # 2. Create a simple answer without LLM
    answer = (
        f"Here is the relevant information from the dataset:\n\n"
        f"{context_text}\n\n"
        f"(Answer generated through retrieval only — no LLM used.)"
    )

    # 3. Extract sources
    sources = [d.metadata.get("source", "unknown") for d in docs]

    return {
        "answer": answer,
        "context": context_text,
        "sources": sources,
        "history": []
    }