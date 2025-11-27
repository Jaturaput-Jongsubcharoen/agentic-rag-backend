from pathlib import Path
# Lazy imports of langchain components are performed inside the functions
# to avoid hard dependency at module import time when langchain isn't installed.

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

def load_documents():
    """
    Loads all .txt docs from /data.
    You can extend this for pdf, md, etc.
    """
    try:
        from langchain.document_loaders import DirectoryLoader, TextLoader
    except Exception as e:
        raise ImportError(
            "langchain is required for loading documents. Install with 'pip install langchain'"
        ) from e

    loader = DirectoryLoader(
        str(DATA_DIR),
        glob="**/*.txt",
        loader_cls=TextLoader,
        show_progress=True,
    )
    docs = loader.load()
    return docs

def load_and_split_documents(chunk_size: int = 1000, chunk_overlap: int = 200):
    docs = load_documents()
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except Exception as e:
        raise ImportError(
            "langchain is required for splitting documents. Install with 'pip install langchain'"
        ) from e
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(docs)
    return chunks