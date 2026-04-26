"""
monitoring/feedback.py
───────────────────────
User feedback storage for RAG answers.

Stores thumbs-up/down feedback as a JSON array in logs/feedback.json.
Uses atomic writes (write to temp file, then rename) to prevent corruption.
"""
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import LOGS_DIR

FEEDBACK_FILE = Path(LOGS_DIR) / "feedback.json"


def save_feedback(
    query: str,
    answer: str,
    feedback: str,
    sources: Optional[List] = None,
) -> None:
    """
    Append a feedback entry to the feedback log.

    Args:
        query:    The user's question.
        answer:   The assistant's response (truncated to 500 chars).
        feedback: "positive" or "negative".
        sources:  Source metadata list (optional).
    """
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query":     query,
        "answer":    answer[:500],
        "feedback":  feedback,
        "sources":   [s.get("filename", "") for s in (sources or [])],
    }

    # Load existing entries
    entries = []
    if FEEDBACK_FILE.exists():
        try:
            with open(FEEDBACK_FILE, encoding="utf-8") as f:
                entries = json.load(f)
        except (json.JSONDecodeError, OSError):
            entries = []

    entries.append(entry)

    # Atomic write: write to temp, then rename
    fd, tmp_path = tempfile.mkstemp(dir=LOGS_DIR, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, str(FEEDBACK_FILE))
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def get_feedback_stats() -> dict:
    """Return basic feedback counts."""
    if not FEEDBACK_FILE.exists():
        return {"positive": 0, "neutral": 0, "negative": 0, "total": 0}
    try:
        with open(FEEDBACK_FILE, encoding="utf-8") as f:
            entries = json.load(f)
        pos = sum(1 for e in entries if e.get("feedback") == "positive")
        neu = sum(1 for e in entries if e.get("feedback") == "neutral")
        neg = sum(1 for e in entries if e.get("feedback") == "negative")
        return {"positive": pos, "neutral": neu, "negative": neg, "total": len(entries)}
    except Exception:
        return {"positive": 0, "neutral": 0, "negative": 0, "total": 0}
