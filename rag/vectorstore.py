from functools import lru_cache
from typing import Any
from langchain_community.vectorstores import FAISS

from .loader import load_and_split_documents
from .embeddings import get_embeddings


@lru_cache(maxsize=1)
def get_vectorstore() -> FAISS:
    """
    Build (or cache) the FAISS vector store.
    Called once at startup and reused.
    """
    chunks = load_and_split_documents()
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


def get_retriever(k: int = 4) -> Any:
    vs = get_vectorstore()
    return vs.as_retriever(search_kwargs={"k": k})
