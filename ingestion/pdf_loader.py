"""
ingestion/pdf_loader.py
────────────────────────
Document loading utilities.
Supports PDF, DOCX, and plain text files via LangChain loaders.
"""
import sys
from pathlib import Path
from typing import List

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.schema import Document

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}
_MIN_CHARS_PER_PAGE = 50   # Below this → extraction is probably broken


# ── OCR / extraction helpers ─────────────────────────────────────────────────

def _is_extraction_poor(documents: List[Document]) -> bool:
    """Return True if text extraction likely failed (scanned/complex PDF)."""
    if not documents:
        return True
    total_chars  = sum(len(d.page_content.strip()) for d in documents)
    avg_per_page = total_chars / len(documents)
    return avg_per_page < _MIN_CHARS_PER_PAGE


def _load_pdf_with_pymupdf(file_path: str) -> List[Document]:
    """
    Fallback PDF loader using PyMuPDF (fitz).
    Handles embedded fonts, rotated text, and complex layouts better than pypdf.
    """
    import fitz  # pymupdf

    docs = []
    filename = Path(file_path).name
    pdf = fitz.open(file_path)
    for page_num, page in enumerate(pdf):
        text = page.get_text("text")
        if text.strip():
            docs.append(Document(
                page_content=text,
                metadata={
                    "source": filename,
                    "file_path": file_path,
                    "page": page_num,
                    "extraction_method": "pymupdf",
                },
            ))
    pdf.close()
    return docs


# ── Single-file loaders ─────────────────────────────────────────────────────

def load_pdf(file_path: str) -> List[Document]:
    """
    Load a single PDF file.
    If standard extraction yields poor results, falls back to PyMuPDF
    for better handling of scanned/complex PDFs.
    """
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    filename = Path(file_path).name
    for doc in documents:
        doc.metadata["source"]    = filename
        doc.metadata["file_path"] = file_path
        doc.metadata["extraction_method"] = "pypdf"

    # Fallback: if pypdf extracted almost no text, try PyMuPDF
    if _is_extraction_poor(documents):
        try:
            fallback_docs = _load_pdf_with_pymupdf(file_path)
            if fallback_docs and not _is_extraction_poor(fallback_docs):
                return fallback_docs
        except ImportError:
            pass   # pymupdf not installed — use whatever pypdf got
        except Exception:
            pass   # any other error — use original

    return documents


def load_docx(file_path: str) -> List[Document]:
    """
    Load a single DOCX file via docx2txt.
    Returns one Document (DOCX has no native page concept).
    """
    from langchain_community.document_loaders import Docx2txtLoader

    loader = Docx2txtLoader(file_path)
    documents = loader.load()
    filename = Path(file_path).name
    for doc in documents:
        doc.metadata["source"]    = filename
        doc.metadata["file_path"] = file_path
        doc.metadata["page"]      = 0
    return documents


def load_text(file_path: str) -> List[Document]:
    """
    Load a plain text file with automatic encoding detection.
    Returns one Document.
    """
    from langchain_community.document_loaders import TextLoader

    loader = TextLoader(file_path, encoding="utf-8", autodetect_encoding=True)
    documents = loader.load()
    filename = Path(file_path).name
    for doc in documents:
        doc.metadata["source"]    = filename
        doc.metadata["file_path"] = file_path
        doc.metadata["page"]      = 0
    return documents


def load_file(file_path: str) -> List[Document]:
    """
    Dispatcher — load any supported file type by extension.
    Raises ValueError for unsupported extensions.
    """
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return load_pdf(file_path)
    if ext == ".docx":
        return load_docx(file_path)
    if ext == ".txt":
        return load_text(file_path)
    raise ValueError(
        f"Unsupported file type '{ext}'. "
        f"Supported: {', '.join(sorted(_SUPPORTED_EXTENSIONS))}"
    )


# ── Directory loader (library ingestion) ─────────────────────────────────────

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
