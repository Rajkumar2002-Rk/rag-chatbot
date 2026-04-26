"""Tests for retrieval/query_classifier.py — rule-based classification."""
from retrieval.query_classifier import classify_query, QueryType


def test_factual_query():
    cfg = classify_query("What is the Transformer architecture?")
    assert cfg.query_type == QueryType.FACTUAL
    assert cfg.top_k == 3
    assert cfg.fetch_k == 10


def test_complex_query_by_signal():
    cfg = classify_query("Explain how attention works in Transformers")
    assert cfg.query_type == QueryType.COMPLEX
    assert cfg.top_k == 7
    assert cfg.fetch_k == 25


def test_complex_query_by_length():
    long_q = "I want to understand all the different ways that neural networks process data"
    cfg = classify_query(long_q)
    assert cfg.query_type == QueryType.COMPLEX


def test_ambiguous_query():
    cfg = classify_query("GPT")
    assert cfg.query_type == QueryType.AMBIGUOUS
    assert "GPT" in cfg.query  # expanded query


def test_keyword_query():
    cfg = classify_query("transformer attention mechanism")
    assert cfg.query_type == QueryType.KEYWORD
    assert cfg.top_k == 5
