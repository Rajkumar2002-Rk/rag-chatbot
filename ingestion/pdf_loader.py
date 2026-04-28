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


def _extract_tables_as_markdown(page) -> str:
    """Extract tables from a page and format as markdown."""
    try:
        tables = page.find_tables()
        if not tables or len(tables.tables) == 0:
            return ""

        parts = []
        for table in tables:
            rows = table.extract()
            if not rows or len(rows) < 1:
                continue
            # Build markdown table
            header = rows[0]
            col_count = len(header)
            header_cells = [str(c or "").strip() for c in header]
            lines = ["| " + " | ".join(header_cells) + " |"]
            lines.append("| " + " | ".join(["---"] * col_count) + " |")
            for row in rows[1:]:
                cells = [str(c or "").strip() for c in row[:col_count]]
                lines.append("| " + " | ".join(cells) + " |")
            parts.append("\n".join(lines))
        return "\n\n".join(parts)
    except Exception:
        return ""


def _count_images(page) -> int:
    """Count meaningful images on a page (skip tiny icons)."""
    try:
        images = page.get_images(full=True)
        # Filter out tiny images (likely icons/bullets)
        count = 0
        for img in images:
            xref = img[0]
            try:
                base_image = page.parent.extract_image(xref)
                if base_image and base_image.get("width", 0) > 50 and base_image.get("height", 0) > 50:
                    count += 1
            except Exception:
                count += 1  # Count it if we can't check size
        return count
    except Exception:
        return 0


def _load_pdf_with_pymupdf(file_path: str) -> List[Document]:
    """
    Fallback PDF loader using PyMuPDF (fitz).
    Handles embedded fonts, rotated text, tables, and complex layouts
    better than pypdf. Tables are extracted as markdown tables.
    Images are noted as placeholders.
    """
    import fitz  # pymupdf

    docs = []
    filename = Path(file_path).name
    pdf = fitz.open(file_path)
    for page_num, page in enumerate(pdf):
        parts = []

        # 1. Extract regular text
        text = page.get_text("text")
        if text.strip():
            parts.append(text.strip())

        # 2. Extract tables as markdown
        table_md = _extract_tables_as_markdown(page)
        if table_md:
            parts.append("\n[TABLE]\n" + table_md + "\n[/TABLE]")

        # 3. Note images/charts present on the page
        img_count = _count_images(page)
        if img_count > 0:
            parts.append(
                f"\n[This page contains {img_count} image(s)/chart(s) "
                f"that cannot be extracted as text.]"
            )

        page_content = "\n\n".join(parts)
        if page_content.strip():
            docs.append(Document(
                page_content=page_content,
                metadata={
                    "source": filename,
                    "file_path": file_path,
                    "page": page_num,
                    "extraction_method": "pymupdf",
                    "has_tables": bool(table_md),
                    "image_count": img_count,
                },
            ))
    pdf.close()
    return docs


# ── Single-file loaders ─────────────────────────────────────────────────────

def load_pdf(file_path: str) -> List[Document]:
    """
    Load a single PDF file.
    Tries PyMuPDF first for better table/layout handling, then falls
    back to pypdf if PyMuPDF is unavailable or fails.
    """
    # Try PyMuPDF first — handles tables, complex layouts, images
    try:
        pymupdf_docs = _load_pdf_with_pymupdf(file_path)
        if pymupdf_docs and not _is_extraction_poor(pymupdf_docs):
            return pymupdf_docs
    except ImportError:
        pass   # pymupdf not installed
    except Exception:
        pass   # any other error

    # Fallback to pypdf
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    filename = Path(file_path).name
    for doc in documents:
        doc.metadata["source"]    = filename
        doc.metadata["file_path"] = file_path
        doc.metadata["extraction_method"] = "pypdf"

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
