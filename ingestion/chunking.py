"""
ingestion/chunking.py
──────────────────────
Adaptive text splitting utilities.

Chunk size adapts to document length so short docs (resumes, 1-pagers)
get fine-grained chunks while long docs keep full-paragraph chunks:

  Document size         chunk_size  overlap
  ─────────────────────────────────────────
  Short  (<3 000 chars)     500       100
  Medium (<20 000 chars)    800       150
  Long   (>= 20 000)      1000       200
"""
import sys
from pathlib import Path
from typing import List, Tuple

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP


def _get_chunk_params(documents: List[Document]) -> Tuple[int, int]:
    """Choose chunk size and overlap based on total document length."""
    total_chars = sum(len(d.page_content) for d in documents)

    if total_chars < 3_000:       # ~1-2 pages (resumes, letters)
        return 500, 100
    if total_chars < 20_000:      # ~5-20 pages
        return 800, 150
    return CHUNK_SIZE, CHUNK_OVERLAP   # long docs — use config defaults


def split_documents(documents: List[Document]) -> List[Document]:
    """
    Split a list of Documents into smaller chunks.

    Chunk size adapts to document length automatically.
    Adds chunk_index to metadata for traceability.
    """
    if not documents:
        return []

    chunk_size, chunk_overlap = _get_chunk_params(documents)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i

    return chunks
