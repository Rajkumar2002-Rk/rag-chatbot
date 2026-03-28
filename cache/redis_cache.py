"""
cache/redis_cache.py
─────────────────────
Redis-backed query result cache.

Cache key  : SHA-256 of (normalized query + sorted filter_docs list)
Cache value: JSON-serialized result dict from run_query()
TTL        : CACHE_TTL seconds (default 1 hour)

Design decisions:
- Lazy connection  — Redis client created on first use, not at import time.
- Fail-silent      — Any Redis error logs a warning and returns None/no-op,
                     so the app continues working without Redis.
- filter_docs in key — same question scoped to different docs must not
                     return the wrong cached answer.
- Skip bad results — fallback, low-confidence, and error responses are
                     never stored; only high-quality answers are cached.
- Normalization    — queries are normalized before hashing so surface
                     variations ("What is X?", "what is x", "what is x.")
                     all map to the same cache entry.
"""
import hashlib
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import REDIS_URL, CACHE_TTL

logger = logging.getLogger(__name__)

_client = None   # module-level Redis singleton

# ── In-process stats counters ─────────────────────────────────────────────────
_stats: Dict[str, int] = {
    "hits":    0,
    "misses":  0,
    "sets":    0,
    "skipped": 0,   # results not cached because they were low-quality
}

# Conversational filler prefixes to strip before hashing.
# Only the first matching prefix is removed.
_FILLER_PREFIXES = (
    "please ",
    "can you please ",
    "can you ",
    "could you please ",
    "could you ",
    "would you ",
    "tell me ",
    "i want to know ",
    "i'd like to know ",
    "hey ",
    "hi ",
)


def _get_client():
    """Return a shared Redis client, creating it on first call."""
    global _client
    if _client is None:
        import redis
        _client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    return _client


def normalize_query(query: str) -> str:
    """
    Normalize a query string so surface variations share one cache entry.

    Steps (in order):
      1. Strip leading/trailing whitespace
      2. Lowercase
      3. Collapse internal whitespace runs to a single space
      4. Strip trailing punctuation (?, !, ., ,, ;)
      5. Strip one leading conversational filler phrase

    Examples:
      "What is RAG?"          → "what is rag"
      "what is rag."          → "what is rag"
      "Can you explain RAG?"  → "explain rag"
      "EXPLAIN  RAG  !"       → "explain rag"
    """
    q = query.strip().lower()
    q = re.sub(r'\s+', ' ', q)       # collapse whitespace
    q = q.rstrip('?!.,;')            # strip trailing punctuation
    for prefix in _FILLER_PREFIXES:
        if q.startswith(prefix):
            q = q[len(prefix):]
            break
    return q.strip()


def _make_key(query: str, filter_docs: Optional[List[str]]) -> str:
    """
    Build a deterministic cache key from the normalized query + filter_docs.

    filter_docs is included so that the same question scoped to different
    document subsets never returns the wrong cached answer.
    """
    normalized = normalize_query(query)
    doc_part   = ",".join(sorted(filter_docs)) if filter_docs else ""
    raw        = f"{normalized}||{doc_part}"
    digest     = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"rag:cache:{digest}"


def _should_cache(result: Dict[str, Any]) -> bool:
    """
    Return True only for high-quality results worth caching.

    Skipped cases:
      - fallback_triggered=True  : guardrail fired, answer is a canned message
      - confidence_ok=False      : retrieval confidence was too low
      - "error" key present      : an exception occurred during query
    """
    if result.get("fallback_triggered"):
        return False
    if result.get("confidence_ok") is False:
        return False
    if result.get("error"):
        return False
    return True


def get_cached_result(
    query: str,
    filter_docs: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Return the cached result dict for this query, or None on miss/error.
    Increments _stats["hits"] or _stats["misses"] accordingly.
    """
    try:
        key  = _make_key(query, filter_docs)
        data = _get_client().get(key)
        if data is None:
            _stats["misses"] += 1
            logger.debug("Cache MISS key=%s", key)
            return None
        _stats["hits"] += 1
        logger.info("Cache HIT  key=%s", key)
        return json.loads(data)
    except Exception as exc:
        _stats["misses"] += 1
        logger.warning("Cache GET failed (Redis unavailable?): %s", exc)
        return None


def set_cached_result(
    query: str,
    result: Dict[str, Any],
    filter_docs: Optional[List[str]] = None,
) -> None:
    """
    Store result in Redis with TTL.

    Low-quality results (fallback, low-confidence, errors) are silently
    skipped — no point serving cached bad answers.
    Fails silently on Redis errors.
    """
    if not _should_cache(result):
        _stats["skipped"] += 1
        logger.debug(
            "Cache SKIP (fallback=%s confidence_ok=%s error=%s)",
            result.get("fallback_triggered"),
            result.get("confidence_ok"),
            result.get("error"),
        )
        return

    try:
        key = _make_key(query, filter_docs)
        _get_client().setex(key, CACHE_TTL, json.dumps(result))
        _stats["sets"] += 1
        logger.info("Cache SET  key=%s  ttl=%ds", key, CACHE_TTL)
    except Exception as exc:
        logger.warning("Cache SET failed (Redis unavailable?): %s", exc)


def get_cache_stats() -> Dict[str, int]:
    """Return a copy of the current in-process cache counters."""
    return dict(_stats)
