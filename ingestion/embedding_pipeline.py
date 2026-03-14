"""
ingestion/embedding_pipeline.py
─────────────────────────────────
End-to-end ingestion: PDFs → chunks → embeddings → ChromaDB.

Two entry points:
  ingest_directory()    — batch ingest for the library (persisted)
  ingest_uploaded_pdf() — single-file ingest for upload mode (in-memory or persisted)
"""
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

from langchain.schema import Document

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vectorstore.chroma_manager import ChromaManager
from ingestion.pdf_loader   import load_directory, load_pdf, patch_metadata_source
from ingestion.chunking     import split_documents
from monitoring.logger      import log_ingestion_event


def ingest_directory(directory: str, collection_name: str = "library") -> int:
    """
    Ingest all PDFs from a directory into a persisted ChromaDB collection.

    Returns:
        Number of chunks indexed.
    """

    t0 = time.time()
    print(f"[ingest] Loading PDFs from: {directory}")
    documents = load_directory(directory)

    if not documents:
        raise ValueError(f"No PDF documents found in: {directory}")

    unique_files = set(d.metadata.get("source", "") for d in documents)
    print(f"[ingest] Loaded {len(documents)} pages from {len(unique_files)} file(s)")

    chunks = split_documents(documents)
    print(f"[ingest] Split into {len(chunks)} chunks")

    manager = ChromaManager(collection_name=collection_name, persist=True)
    manager.add_documents(chunks)

    duration = time.time() - t0
    for filename in unique_files:
        file_chunks = [c for c in chunks if c.metadata.get("source") == filename]
        log_ingestion_event(filename, len(file_chunks), duration)

    print(f"[ingest] Done — {len(chunks)} chunks indexed into '{collection_name}' ({duration:.1f}s)")
    return len(chunks)


def ingest_uploaded_pdf(
    file_path: str,
    filename: str,
    collection_name: str = "upload_session",
    persist: bool = False,
) -> Tuple[List[Document], "ChromaManager"]:  # type: ignore[name-defined]
    """
    Ingest a single uploaded PDF.

    Args:
        file_path:       Absolute path to the temp file on disk.
        filename:        The user-visible filename (e.g. "China - Wikipedia.pdf").
        collection_name: ChromaDB collection name for this session.
        persist:         If False, uses in-memory ChromaDB (nothing saved to disk).

    Returns:
        (chunks, ChromaManager) — the chunks for display and the store for querying.
    """

    t0 = time.time()
    documents = load_pdf(file_path)
    documents = patch_metadata_source(documents, filename)
    chunks    = split_documents(documents)

    manager = ChromaManager(collection_name=collection_name, persist=persist)
    manager.add_documents(chunks)

    log_ingestion_event(filename, len(chunks), time.time() - t0)
    return chunks, manager
