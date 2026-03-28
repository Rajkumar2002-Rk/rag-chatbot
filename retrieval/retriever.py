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
    "I could not find relevant information in the uploaded document."
)


# ── MMR Retriever ─────────────────────────────────────────────────────────────

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
    search_kwargs: dict = {
        "k":           k,
        "fetch_k":     fetch_k,
        "lambda_mult": lambda_mult,
    }
    if filter_docs:
        if len(filter_docs) == 1:
            search_kwargs["filter"] = {"source": filter_docs[0]}
        elif len(filter_docs) > 1:
            search_kwargs["filter"] = {"source": {"$in": filter_docs}}

    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs=search_kwargs,
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
    kwargs: dict = {"k": k}
    if filter_docs:
        if len(filter_docs) == 1:
            kwargs["filter"] = {"source": filter_docs[0]}
        elif len(filter_docs) > 1:
            kwargs["filter"] = {"source": {"$in": filter_docs}}

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

    Three guardrail conditions:
      1. No documents retrieved at all
      2. Best similarity score is below SIMILARITY_THRESHOLD
      3. Total context is too short to be meaningful

    Returns:
        (is_confident, reason_code)
        reason_code is "ok" when confident, else a descriptive failure code.
    """
    if not docs:
        return False, "no_documents_retrieved"

    best_score = max(scores) if scores else 0.0
    if best_score < SIMILARITY_THRESHOLD:
        return False, f"low_confidence_score:{best_score:.3f}_threshold:{SIMILARITY_THRESHOLD}"

    total_context = " ".join(doc.page_content for doc in docs)
    if len(total_context.strip()) < MIN_CONTEXT_LENGTH:
        return False, "insufficient_context_length"

    return True, "ok"
