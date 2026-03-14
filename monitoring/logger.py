"""
monitoring/logger.py
─────────────────────
Structured JSON logger for the RAG pipeline.
Every query event is written to logs/rag_queries.log as a single JSON line.

Log fields:
    timestamp, level, message, query, response_preview,
    response_time_ms, retrieval_time_ms, num_chunks,
    source_documents, token_usage_estimate, error
"""
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

# ── Bootstrap path so this module can be imported from any working directory ──
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import LOGS_DIR

LOG_FILE = Path(LOGS_DIR) / "rag_queries.log"


class _JSONFormatter(logging.Formatter):
    """Formats log records as compact single-line JSON strings."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        # Merge any extra fields attached to the record
        if hasattr(record, "extra"):
            payload.update(record.extra)
        return json.dumps(payload, ensure_ascii=False)


def _build_logger() -> logging.Logger:
    logger = logging.getLogger("rag_system")
    if logger.handlers:          # Avoid duplicate handlers on Streamlit reruns
        return logger
    logger.setLevel(logging.INFO)

    # File handler — JSON lines (machine-readable)
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setFormatter(_JSONFormatter())
    logger.addHandler(fh)

    # Console handler — human-readable
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(ch)

    return logger


logger = _build_logger()


# ── Public API ────────────────────────────────────────────────────────────────

def log_query_event(
    query: str,
    response: str,
    response_time: float,
    retrieval_time: float,
    num_chunks: int,
    source_documents: List[str],
    token_usage: Optional[int] = None,
    error: Optional[str] = None,
) -> None:
    """
    Log a complete query lifecycle event.

    Args:
        query:            The user's question.
        response:         The LLM's answer (or fallback string).
        response_time:    End-to-end wall time in seconds.
        retrieval_time:   ChromaDB retrieval wall time in seconds.
        num_chunks:       Number of chunks passed to the LLM.
        source_documents: List of source filenames used.
        token_usage:      Estimated token count (prompt + completion).
        error:            Error description if the query failed.
    """
    extra = {
        "query":                query,
        "response_preview":     response[:200] if response else "",
        "response_time_ms":     round(response_time   * 1000, 2),
        "retrieval_time_ms":    round(retrieval_time  * 1000, 2),
        "num_chunks":           num_chunks,
        "source_documents":     source_documents,
        "token_usage_estimate": token_usage,
        "error":                error,
    }
    record = logging.LogRecord(
        name="rag_system", level=logging.INFO,
        pathname="", lineno=0,
        msg="QUERY_EVENT", args=(), exc_info=None,
    )
    record.extra = extra  # type: ignore[attr-defined]
    logger.handle(record)


def log_ingestion_event(filename: str, num_chunks: int, duration: float) -> None:
    """Log a document ingestion event."""
    extra = {
        "filename":       filename,
        "chunks_indexed": num_chunks,
        "duration_ms":    round(duration * 1000, 2),
    }
    record = logging.LogRecord(
        name="rag_system", level=logging.INFO,
        pathname="", lineno=0,
        msg="INGESTION_EVENT", args=(), exc_info=None,
    )
    record.extra = extra  # type: ignore[attr-defined]
    logger.handle(record)
