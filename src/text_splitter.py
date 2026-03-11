import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP


def split_documents(documents: list, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> list:
    """
    Split documents into smaller chunks for embedding.

    Args:
        documents: List of LangChain Document objects
        chunk_size: Max characters per chunk (default from config: 1000)
        chunk_overlap: Overlap between chunks (default from config: 200)

    Returns:
        List of chunked Document objects
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True
    )

    chunks = splitter.split_documents(documents)

    print(f"Split {len(documents)} pages into {len(chunks)} chunks")
    print(f"  chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")

    return chunks
