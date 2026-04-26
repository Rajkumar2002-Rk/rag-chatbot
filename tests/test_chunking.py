"""Tests for ingestion/chunking.py — adaptive chunk sizing."""
from ingestion.chunking import _get_chunk_params, split_documents


def test_short_document_uses_small_chunks(short_documents):
    chunk_size, overlap = _get_chunk_params(short_documents)
    assert chunk_size == 500
    assert overlap == 100


def test_medium_document_uses_medium_chunks(medium_documents):
    chunk_size, overlap = _get_chunk_params(medium_documents)
    assert chunk_size == 800
    assert overlap == 150


def test_long_document_uses_large_chunks(long_documents):
    chunk_size, overlap = _get_chunk_params(long_documents)
    assert chunk_size == 1000
    assert overlap == 200


def test_split_documents_adds_chunk_index(short_documents):
    chunks = split_documents(short_documents)
    assert len(chunks) > 0
    for i, chunk in enumerate(chunks):
        assert chunk.metadata["chunk_index"] == i


def test_split_documents_empty_input():
    assert split_documents([]) == []
