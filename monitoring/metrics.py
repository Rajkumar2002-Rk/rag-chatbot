"""
monitoring/metrics.py
──────────────────────
In-memory metrics tracker for the Streamlit dashboard.

Stored in st.session_state so it persists across reruns within a session.
Tracks query counts, latencies, token usage, and document access patterns.
"""
from collections import defaultdict
from typing import Dict, List, Tuple


class MetricsTracker:
    """
    Lightweight metrics collector — no external dependencies.

    Usage (in Streamlit):
        if "metrics" not in st.session_state:
            st.session_state.metrics = MetricsTracker()
        st.session_state.metrics.record_query(...)
    """

    def __init__(self) -> None:
        self.total_queries:       int        = 0
        self.total_tokens:        int        = 0
        self.response_times:      List[float] = []
        self.retrieval_times:     List[float] = []
        self.document_access:     Dict[str, int] = defaultdict(int)
        self.failed_queries:      int        = 0
        self.empty_context_count: int        = 0
        self.fallback_count:      int        = 0

    # ── Record ────────────────────────────────────────────────────────────────

    def record_query(
        self,
        response_time:    float,
        retrieval_time:   float,
        source_documents: List[str],
        token_usage:      int  = 0,
        failed:           bool = False,
        empty_context:    bool = False,
        fallback:         bool = False,
    ) -> None:
        self.total_queries  += 1
        self.response_times.append(response_time)
        self.retrieval_times.append(retrieval_time)
        self.total_tokens   += token_usage

        for doc in source_documents:
            self.document_access[doc] += 1

        if failed:       self.failed_queries      += 1
        if empty_context: self.empty_context_count += 1
        if fallback:     self.fallback_count       += 1

    # ── Computed properties ───────────────────────────────────────────────────

    @property
    def avg_response_time_ms(self) -> float:
        if not self.response_times:
            return 0.0
        return round(sum(self.response_times) / len(self.response_times) * 1000, 1)

    @property
    def avg_retrieval_time_ms(self) -> float:
        if not self.retrieval_times:
            return 0.0
        return round(sum(self.retrieval_times) / len(self.retrieval_times) * 1000, 1)

    @property
    def p95_response_time_ms(self) -> float:
        """95th percentile response time in ms."""
        if len(self.response_times) < 2:
            return self.avg_response_time_ms
        sorted_times = sorted(self.response_times)
        idx = int(len(sorted_times) * 0.95)
        return round(sorted_times[idx] * 1000, 1)

    @property
    def top_documents(self) -> List[Tuple[str, int]]:
        """Top 5 most queried documents."""
        return sorted(self.document_access.items(), key=lambda x: x[1], reverse=True)[:5]

    @property
    def success_rate(self) -> float:
        if self.total_queries == 0:
            return 100.0
        return round((1 - self.failed_queries / self.total_queries) * 100, 1)

    # ── Serialization ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "total_queries":           self.total_queries,
            "avg_response_time_ms":    self.avg_response_time_ms,
            "avg_retrieval_time_ms":   self.avg_retrieval_time_ms,
            "p95_response_time_ms":    self.p95_response_time_ms,
            "total_tokens_used":       self.total_tokens,
            "failed_queries":          self.failed_queries,
            "fallback_responses":      self.fallback_count,
            "empty_context_responses": self.empty_context_count,
            "success_rate_pct":        self.success_rate,
            "top_documents":           self.top_documents,
        }
