import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from config import EMBEDDING_MODEL, VECTORSTORE_DIR

load_dotenv()


def get_embeddings():
    """Create and return an OpenAIEmbeddings instance with API key validation."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env file. Please add it.")
    return OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=api_key)


def create_vector_store(chunks: list, persist_directory: str = VECTORSTORE_DIR):
    """
    Create a new ChromaDB vector store from document chunks.

    Args:
        chunks: List of chunked Document objects
        persist_directory: Folder to save the vector store

    Returns:
        Chroma vector store instance
    """
    try:
        embeddings = get_embeddings()
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        print(f"Vector store created with {len(chunks)} chunks at '{persist_directory}'")
        return vector_store

    except Exception as e:
        error_str = str(e).lower()
        if "quota" in error_str or "billing" in error_str or "insufficient" in error_str:
            raise RuntimeError(
                "OpenAI quota exceeded. Please add credits at: platform.openai.com/billing"
            ) from e
        raise


def load_vector_store(persist_directory: str = VECTORSTORE_DIR):
    """
    Load an existing ChromaDB vector store from disk.

    Args:
        persist_directory: Folder where the vector store was saved

    Returns:
        Chroma vector store instance
    """
    if not os.path.exists(persist_directory):
        raise FileNotFoundError(
            f"Vector store not found at '{persist_directory}'. "
            f"Please run `python ingest.py` first to build it."
        )
    embeddings = get_embeddings()
    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    print(f"Vector store loaded from '{persist_directory}'")
    return vector_store
