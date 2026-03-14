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
    Filter documents below threshold and return top_k by descending score.

    Args:
        docs:      Retrieved documents.
        scores:    Corresponding relevance scores [0, 1].
        top_k:     Maximum documents to return.
        threshold: Minimum score to keep a document.

    Returns:
        (filtered_docs, filtered_scores) sorted by descending score.
    """
    paired = [
        (doc, score)
        for doc, score in zip(docs, scores)
        if score >= threshold
    ]
    paired.sort(key=lambda x: x[1], reverse=True)
    paired = paired[:top_k]

    if not paired:
        return [], []

    out_docs, out_scores = zip(*paired)
    return list(out_docs), list(out_scores)
