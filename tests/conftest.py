"""Shared fixtures for the RAG chatbot test suite."""
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain.schema import Document

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def short_documents():
    """Documents totaling < 3000 chars (resume-sized)."""
    return [
        Document(page_content="A" * 1200, metadata={"source": "resume.pdf", "page": 0}),
        Document(page_content="B" * 800, metadata={"source": "resume.pdf", "page": 1}),
    ]


@pytest.fixture
def medium_documents():
    """Documents totaling 3000–20000 chars."""
    return [
        Document(page_content="C" * 5000, metadata={"source": "report.pdf", "page": i})
        for i in range(3)
    ]


@pytest.fixture
def long_documents():
    """Documents totaling > 20000 chars."""
    return [
        Document(page_content="D" * 8000, metadata={"source": "paper.pdf", "page": i})
        for i in range(5)
    ]


@pytest.fixture
def sample_docs_with_scores():
    """Documents with various scores for reranker testing."""
    docs = [
        Document(page_content="Alpha content", metadata={"source": "a.pdf", "page": 0}),
        Document(page_content="Beta content", metadata={"source": "b.pdf", "page": 1}),
        Document(page_content="Gamma content", metadata={"source": "a.pdf", "page": 0}),  # duplicate key
        Document(page_content="Delta content", metadata={"source": "c.pdf", "page": 2}),
        Document(page_content="Epsilon content", metadata={"source": "d.pdf", "page": 0}),
    ]
    scores = [0.85, 0.42, 0.90, 0.08, 0.55]
    return docs, scores


@pytest.fixture
def mock_vector_store():
    """Mock Chroma vector store."""
    store = MagicMock()
    store._collection.count.return_value = 20
    return store
