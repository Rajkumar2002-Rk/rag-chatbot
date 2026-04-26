"""Tests for retrieval/reranker.py — score-based reranking."""
from langchain.schema import Document
from retrieval.reranker import rerank_by_score


def test_sorts_by_descending_score(sample_docs_with_scores):
    docs, scores = sample_docs_with_scores
    out_docs, out_scores = rerank_by_score(docs, scores, top_k=10, threshold=0.0)
    assert out_scores == sorted(out_scores, reverse=True)


def test_filters_below_threshold():
    docs = [
        Document(page_content="good", metadata={"source": "a.pdf", "page": 0}),
        Document(page_content="bad", metadata={"source": "b.pdf", "page": 0}),
    ]
    scores = [0.50, 0.05]
    out_docs, out_scores = rerank_by_score(docs, scores, top_k=5, threshold=0.10)
    assert len(out_docs) == 1
    assert out_scores[0] == 0.50


def test_safety_net_keeps_best_when_all_below_threshold():
    docs = [
        Document(page_content="low1", metadata={"source": "a.pdf", "page": 0}),
        Document(page_content="low2", metadata={"source": "b.pdf", "page": 1}),
    ]
    scores = [0.05, 0.03]
    out_docs, out_scores = rerank_by_score(docs, scores, top_k=3, threshold=0.10)
    # Safety net: should return the best available instead of empty
    assert len(out_docs) == 2
    assert out_scores[0] >= out_scores[1]


def test_deduplicates_same_source_page():
    docs = [
        Document(page_content="version A", metadata={"source": "x.pdf", "page": 1}),
        Document(page_content="version B", metadata={"source": "x.pdf", "page": 1}),
        Document(page_content="different", metadata={"source": "y.pdf", "page": 0}),
    ]
    scores = [0.80, 0.90, 0.70]
    out_docs, out_scores = rerank_by_score(docs, scores, top_k=5, threshold=0.0)
    # Should keep the higher-scored duplicate (0.90) and the other doc
    assert len(out_docs) == 2
    assert out_scores[0] == 0.90


def test_respects_top_k():
    docs = [
        Document(page_content=f"doc{i}", metadata={"source": f"{i}.pdf", "page": 0})
        for i in range(10)
    ]
    scores = [0.9 - i * 0.05 for i in range(10)]
    out_docs, _ = rerank_by_score(docs, scores, top_k=3, threshold=0.0)
    assert len(out_docs) == 3


def test_empty_input():
    docs, scores = rerank_by_score([], [], top_k=5, threshold=0.10)
    assert docs == []
    assert scores == []
