"""
retrieval/retriever.py
───────────────────────
MMR retriever with similarity scoring and hallucination guardrails.

Key responsibilities:
  1. get_mmr_retriever()         — builds LangChain MMR retriever
  2. retrieve_with_scores()      — returns (docs, scores) for guardrail evaluation
  3. check_retrieval_confidence() — decides whether to call the LLM or return fallback
"""
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from langchain.schema import Document
from langchain_chroma import Chroma

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import (
    RETRIEVAL_K, RETRIEVAL_FETCH_K, RETRIEVAL_LAMBDA,
    SIMILARITY_THRESHOLD, MIN_CONTEXT_LENGTH,
)

FALLBACK_RESPONSE = (
    "I could not find relevant information in the provided documents."
)


# ── MMR Retriever ─────────────────────────────────────────────────────────────

def _cap_to_collection(vector_store: Chroma, k: int, fetch_k: int):
    """Cap k and fetch_k so they never exceed the actual number of chunks."""
    try:
        n = vector_store._collection.count()
        if n == 0:
            return k, fetch_k
        k       = min(k, n)
        fetch_k = min(fetch_k, n)
        # fetch_k must be >= k for MMR to work
        fetch_k = max(fetch_k, k)
    except Exception:
        pass
    return k, fetch_k


def _build_filter(filter_docs: Optional[List[str]]) -> Optional[dict]:
    """Build a ChromaDB where-clause from a list of filenames."""
    if not filter_docs:
        return None
    if len(filter_docs) == 1:
        return {"source": {"$eq": filter_docs[0]}}
    return {"source": {"$in": filter_docs}}


def get_mmr_retriever(
    vector_store: Chroma,
    filter_docs: Optional[List[str]] = None,
    k: int = RETRIEVAL_K,
    fetch_k: int = RETRIEVAL_FETCH_K,
    lambda_mult: float = RETRIEVAL_LAMBDA,
):
    """
    Build an MMR retriever from a Chroma store.

    WHY MMR over cosine similarity:
      Standard cosine returns top-k most similar chunks — often near-duplicates
      from the same paragraph. MMR fetches fetch_k=20 candidates and re-ranks
      them to maximize both relevance AND diversity, giving the LLM broader context.

    Args:
        filter_docs:  Restrict retrieval to these source filenames (multi-doc support).
        k:            Number of chunks to return (overrides RETRIEVAL_K).
        fetch_k:      Candidate pool size for MMR (overrides RETRIEVAL_FETCH_K).
        lambda_mult:  MMR diversity weight (overrides RETRIEVAL_LAMBDA).
    """
    k, fetch_k = _cap_to_collection(vector_store, k, fetch_k)

    search_kwargs: dict = {
        "k":           k,
        "fetch_k":     fetch_k,
        "lambda_mult": lambda_mult,
    }
    where = _build_filter(filter_docs)
    if where:
        search_kwargs["filter"] = where

    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs=search_kwargs,
    )


# ── Hybrid Retriever (BM25 + Vector) ─────────────────────────────────────────

def get_hybrid_retriever(
    vector_store: Chroma,
    all_docs: List[Document],
    filter_docs: Optional[List[str]] = None,
    k: int = RETRIEVAL_K,
    fetch_k: int = RETRIEVAL_FETCH_K,
    lambda_mult: float = RETRIEVAL_LAMBDA,
):
    """
    Build an ensemble retriever that combines keyword (BM25) and semantic (MMR).

    BM25 catches exact keyword matches (names, numbers, emails) that
    embeddings sometimes miss. MMR handles semantic similarity. The
    ensemble merges both result sets with reciprocal rank fusion.

    Falls back to MMR-only if all_docs is empty or too small.
    """
    from langchain_community.retrievers import BM25Retriever
    from langchain.retrievers import EnsembleRetriever

    mmr = get_mmr_retriever(vector_store, filter_docs, k, fetch_k, lambda_mult)

    # Filter corpus for BM25 if filter_docs is set
    corpus = all_docs
    if filter_docs:
        filter_set = set(filter_docs)
        corpus = [d for d in all_docs if d.metadata.get("source") in filter_set]

    if len(corpus) < 2:
        return mmr

    bm25 = BM25Retriever.from_documents(corpus, k=k)

    return EnsembleRetriever(
        retrievers=[bm25, mmr],
        weights=[0.3, 0.7],
    )


# ── Scored Retrieval ──────────────────────────────────────────────────────────

def retrieve_with_scores(
    vector_store: Chroma,
    query: str,
    k: int = RETRIEVAL_K,
    filter_docs: Optional[List[str]] = None,
) -> Tuple[List[Document], List[float]]:
    """
    Retrieve documents with cosine relevance scores.

    Returns:
        (documents, scores) — scores in [0, 1], higher = more relevant.
    """
    k, _ = _cap_to_collection(vector_store, k, k)
    kwargs: dict = {"k": k}
    where = _build_filter(filter_docs)
    if where:
        kwargs["filter"] = where

    results = vector_store.similarity_search_with_relevance_scores(query, **kwargs)
    if not results:
        return [], []

    docs, scores = zip(*results)
    return list(docs), list(scores)


# ── Hallucination Guardrails ──────────────────────────────────────────────────

def check_retrieval_confidence(
    docs: List[Document],
    scores: List[float],
) -> Tuple[bool, str]:
    """
    Decide whether retrieved context is trustworthy enough to pass to the LLM.

    Guardrail conditions:
      1. No documents retrieved at all
      2. Total context is too short to be meaningful

    Note: score-based filtering is handled by the reranker, which already
    drops low-scoring chunks while keeping a safety net. The guardrail
    here only blocks truly empty or degenerate results.

    Returns:
        (is_confident, reason_code)
        reason_code is "ok" when confident, else a descriptive failure code.
    """
    if not docs:
        return False, "no_documents_retrieved"

    total_context = " ".join(doc.page_content for doc in docs)
    if len(total_context.strip()) < MIN_CONTEXT_LENGTH:
        return False, "insufficient_context_length"

    return True, "ok"
