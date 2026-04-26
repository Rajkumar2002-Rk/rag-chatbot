"""Integration tests — full pipeline with mocked externals."""
from unittest.mock import patch, MagicMock
from langchain.schema import Document

from api.rag_api import prepare_rag_context


def _make_docs():
    return [
        Document(page_content="Raj Kumar is a software engineer with Python skills.",
                 metadata={"source": "resume.pdf", "page": 0}),
        Document(page_content="He has experience with machine learning and NLP.",
                 metadata={"source": "resume.pdf", "page": 1}),
    ]


@patch("api.rag_api.get_cached_result", return_value=None)
@patch("api.rag_api.retrieve_with_scores")
def test_prepare_context_success(mock_retrieve, mock_cache):
    docs = _make_docs()
    mock_retrieve.return_value = (docs, [0.85, 0.72])

    mock_store = MagicMock()
    mock_store._collection.count.return_value = 10

    result = prepare_rag_context(
        vector_store=mock_store,
        query="What are Raj's skills?",
    )

    assert result["ready"] is True
    assert result["fallback_triggered"] is False
    assert result["num_chunks"] == 2
    assert len(result["sources"]) == 2
    assert "resume.pdf" in result["sources"][0]["filename"]
    assert len(result["context"]) > 0


@patch("api.rag_api.get_cached_result", return_value=None)
@patch("api.rag_api.retrieve_with_scores", return_value=([], []))
def test_prepare_context_fallback_on_empty(mock_retrieve, mock_cache):
    mock_store = MagicMock()
    mock_store._collection.count.return_value = 0

    result = prepare_rag_context(
        vector_store=mock_store,
        query="What is the weather on Mars?",
    )

    assert result["ready"] is False
    assert result["fallback_triggered"] is True
