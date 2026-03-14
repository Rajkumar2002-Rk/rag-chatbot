"""
ingestion/pdf_loader.py
────────────────────────
PDF loading utilities.
Handles single-file and directory-level loading via LangChain's PyPDFLoader.
"""
import sys
from pathlib import Path
from typing import List

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.schema import Document

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_pdf(file_path: str) -> List[Document]:
    """
    Load a single PDF file.
    Normalizes doc.metadata["source"] to just the filename (not full path).
    """
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    filename = Path(file_path).name
    for doc in documents:
        doc.metadata["source"]    = filename
        doc.metadata["file_path"] = file_path
    return documents


def load_directory(directory: str) -> List[Document]:
    """
    Load all PDFs from a directory (recursive).
    Normalizes source metadata to filename only.
    """
    loader = DirectoryLoader(
        directory,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
    )
    documents = loader.load()
    for doc in documents:
        source = doc.metadata.get("source", "")
        doc.metadata["source"] = Path(source).name
    return documents


def patch_metadata_source(documents: List[Document], filename: str) -> List[Document]:
    """
    Replace temp file paths with the real uploaded filename.

    WHY: PyPDFLoader stores the temp path (e.g. /tmp/tmpXXXXX.pdf) as
    doc.metadata["source"]. Without this, citations show the temp name.
    """
    for doc in documents:
        doc.metadata["source"] = filename
    return documents
