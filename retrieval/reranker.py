"""
retrieval/reranker.py
──────────────────────
Score-based reranker.

Filters chunks below the similarity threshold and returns the top-k
by relevance score. This is a lightweight reranker — no cross-encoder
required. Can be upgraded to a cross-encoder (e.g. sentence-transformers)
for higher accuracy in a future iteration.
"""
import sys
from pathlib import Path
from typing import List, Tuple

from langchain.schema import Document

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import SIMILARITY_THRESHOLD, RETRIEVAL_K


def rerank_by_score(
    docs:      List[Document],
    scores:    List[float],
    top_k:     int   = RETRIEVAL_K,
    threshold: float = SIMILARITY_THRESHOLD,
) -> Tuple[List[Document], List[float]]:
    """
    Filter, deduplicate, and return top_k chunks by descending score.

    Steps:
      1. Sort by score descending
      2. Drop chunks below threshold (but keep at least the best chunk
         so the pipeline never returns empty when content was retrieved)
      3. Deduplicate: keep only the highest-scoring chunk per (source, page)
      4. Return top_k

    Deduplication prevents the LLM from seeing the same page content
    multiple times, which wastes context and produces redundant citations.

    Args:
        docs:      Retrieved documents.
        scores:    Corresponding relevance scores [0, 1].
        top_k:     Maximum documents to return.
        threshold: Minimum score to keep a document.

    Returns:
        (filtered_docs, filtered_scores) sorted by descending score.
    """
    if not docs:
        return [], []

    # Sort all pairs by descending score first
    all_paired = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

    # Filter by threshold
    paired = [(doc, score) for doc, score in all_paired if score >= threshold]

    # Safety net: if threshold filtered everything, keep the best chunks.
    # A low-scoring chunk is still better than returning nothing when the
    # user is asking about a document that IS in the collection.
    if not paired and all_paired:
        paired = all_paired[:top_k]

    # Deduplicate: one chunk per (source, page) — keep highest score
    seen: set = set()
    deduped = []
    for doc, score in paired:
        key = (
            doc.metadata.get("source", ""),
            doc.metadata.get("page", ""),
        )
        if key not in seen:
            seen.add(key)
            deduped.append((doc, score))

    deduped = deduped[:top_k]

    out_docs, out_scores = zip(*deduped)
    return list(out_docs), list(out_scores)
