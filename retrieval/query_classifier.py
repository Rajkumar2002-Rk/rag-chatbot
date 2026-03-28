"""
retrieval/query_classifier.py
──────────────────────────────
Rule-based query classifier.

Classifies a query into one of four types and returns a RetrievalConfig
that tells run_query() how many chunks to fetch and which query string
to use for vector search.

Types and retrieval behaviour:
  FACTUAL   — "What is X?" / "Who was Y?"
               top_k=3, fetch_k=10  (precise, fewer chunks needed)
  COMPLEX   — long queries or explanation/comparison requests
               top_k=7, fetch_k=25  (needs more context)
  AMBIGUOUS — 1-2 word queries with no clear intent
               top_k=5, fetch_k=15, query expanded to "What is X? Explain X."
  KEYWORD   — 3-10 word noun phrases with no question structure
               top_k=5, fetch_k=15  (broad match)

No LLM is used — all classification is done with string matching.
"""
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List

from config.settings import RETRIEVAL_LAMBDA


class QueryType(str, Enum):
    FACTUAL   = "factual"
    KEYWORD   = "keyword"
    AMBIGUOUS = "ambiguous"
    COMPLEX   = "complex"


@dataclass
class RetrievalConfig:
    query_type:  QueryType
    query:       str          # original or expanded query used for vector search
    top_k:       int
    fetch_k:     int
    lambda_mult: float = field(default_factory=lambda: RETRIEVAL_LAMBDA)


# ── Signals ───────────────────────────────────────────────────────────────────

_COMPLEX_SIGNALS: List[str] = [
    "explain", "compare", "difference", "differences", "relationship",
    "why", "how does", "how do", "how did", "summarize", "summarise",
    "summary", "describe", "contrast", "analyze", "analyse",
    "elaborate", "discuss", "overview", "tell me about",
    "advantages", "disadvantages", "pros and cons",
    "what are the", "how would", "what happens",
]

_FACTUAL_STARTS: tuple = (
    "what is", "what are", "what was", "what were",
    "who is",  "who was",  "who are",  "who were",
    "when did", "when was", "when were", "when is",
    "where is", "where was", "where are",
    "which", "how many", "how much",
    "does ", "did ", "is ", "are ", "was ", "were ",
)


# ── Classifier ────────────────────────────────────────────────────────────────

def classify_query(query: str) -> RetrievalConfig:
    """
    Classify query and return a RetrievalConfig with appropriate parameters.

    Decision order:
      1. COMPLEX  — long (>10 words) or contains complexity signal words
      2. FACTUAL  — starts with a factual question word
      3. AMBIGUOUS — 1-2 words (no clear intent)
      4. KEYWORD  — everything else (3-10 word noun phrase)
    """
    q     = query.strip().lower()
    # collapse whitespace so multi-word signals match reliably
    q     = re.sub(r'\s+', ' ', q)
    words = q.split()
    n     = len(words)

    # ── COMPLEX ───────────────────────────────────────────────────────────────
    if n > 10 or any(signal in q for signal in _COMPLEX_SIGNALS):
        return RetrievalConfig(
            query_type=QueryType.COMPLEX,
            query=query,
            top_k=7,
            fetch_k=25,
        )

    # ── FACTUAL ───────────────────────────────────────────────────────────────
    if any(q.startswith(start) for start in _FACTUAL_STARTS):
        return RetrievalConfig(
            query_type=QueryType.FACTUAL,
            query=query,
            top_k=3,
            fetch_k=10,
        )

    # ── AMBIGUOUS ─────────────────────────────────────────────────────────────
    if n <= 2:
        expanded = f"What is {query}? Explain {query} in detail."
        return RetrievalConfig(
            query_type=QueryType.AMBIGUOUS,
            query=expanded,
            top_k=5,
            fetch_k=15,
        )

    # ── KEYWORD ───────────────────────────────────────────────────────────────
    return RetrievalConfig(
        query_type=QueryType.KEYWORD,
        query=query,
        top_k=5,
        fetch_k=15,
    )
