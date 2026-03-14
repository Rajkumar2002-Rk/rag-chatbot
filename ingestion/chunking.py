"""
ingestion/chunking.py
──────────────────────
Text splitting utilities.

Why 1000 / 200:
  - chunk_size=1000  preserves full paragraphs and complete ideas
  - chunk_overlap=200 (20%) ensures no information is lost at chunk boundaries
  - Industry-standard for RAG pipelines on prose documents
"""
import sys
from pathlib import Path
from typing import List

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP


def split_documents(documents: List[Document]) -> List[Document]:
    """
    Split a list of Documents into smaller chunks.
    Adds chunk_index to metadata for traceability.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(documents)

    # Tag each chunk with its position for evaluation and debugging
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i

    return chunks
