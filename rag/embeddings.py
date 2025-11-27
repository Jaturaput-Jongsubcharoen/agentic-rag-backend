import os
import importlib
from dotenv import load_dotenv

load_dotenv()  # load GOOGLE_API_KEY from .env


def get_embeddings():
    """
    Returns a Gemini embedding model instance.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not set in .env")

    try:
        module = importlib.import_module("langchain_google_genai")
        GoogleGenerativeAIEmbeddings = getattr(module, "GoogleGenerativeAIEmbeddings")
    except (ImportError, ModuleNotFoundError, AttributeError):
        raise ImportError(
            "The 'langchain_google_genai' package or 'GoogleGenerativeAIEmbeddings' class is not available; "
            "install it with: pip install langchain-google-genai"
        )

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004"
    )
    return embeddings
