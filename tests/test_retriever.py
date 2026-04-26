"""Tests for retrieval/retriever.py — filter builder and capping."""
from retrieval.retriever import _build_filter


def test_none_returns_none():
    assert _build_filter(None) is None
    assert _build_filter([]) is None


def test_single_doc_uses_eq():
    result = _build_filter(["resume.pdf"])
    assert result == {"source": {"$eq": "resume.pdf"}}


def test_multiple_docs_uses_in():
    result = _build_filter(["a.pdf", "b.pdf"])
    assert result == {"source": {"$in": ["a.pdf", "b.pdf"]}}
