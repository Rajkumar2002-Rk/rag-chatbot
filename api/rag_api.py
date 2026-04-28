"""
api/rag_api.py
───────────────
Core RAG logic — completely decoupled from Streamlit.

Public API:
    format_docs_with_metadata(docs)                         → formatted context string
    prepare_rag_context(vector_store, query, ...)           → preparation dict
    stream_rag_response(prepared, query)                    → generator of string tokens
    run_query(vector_store, query, ...)                     → structured result dict (non-streaming)
"""
import sys
import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain.schema import Document
from langchain_chroma import Chroma

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import OPENAI_API_KEY, LLM_MODEL, LLM_TEMPERATURE, PROMPTS_DIR, SIMILARITY_THRESHOLD
from retrieval.retriever import (
    FALLBACK_RESPONSE,
    check_retrieval_confidence,
    get_mmr_retriever,
    get_hybrid_retriever,
    retrieve_with_scores,
)
from retrieval.reranker          import rerank_by_score
from retrieval.query_classifier  import classify_query
from monitoring.logger           import log_query_event
from cache.redis_cache           import get_cached_result, set_cached_result


# ── Prompt loading ────────────────────────────────────────────────────────────

def _load_prompt_template() -> str:
    """Load prompt from prompts/rag_prompt.txt with inline fallback."""
    prompt_path = Path(PROMPTS_DIR) / "rag_prompt.txt"
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8")
    return (
        "You are a precise document assistant.\n\n"
        "CONTEXT:\n{context}\n\n"
        "CONVERSATION HISTORY:\n{chat_history}\n\n"
        "Question: {question}\n\nANSWER:"
    )


# ── Chat history formatting ──────────────────────────────────────────────────

def _format_chat_history(messages: Optional[List[Dict[str, str]]]) -> str:
    """
    Format the last 3 exchanges into a compact history string.
    Truncates assistant answers to avoid blowing up the context window.
    """
    if not messages:
        return "None"

    # Take last 6 messages (3 user + 3 assistant turns)
    recent = messages[-6:]
    lines = []
    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"]
        # Truncate long assistant answers
        if role == "Assistant" and len(content) > 200:
            content = content[:200] + "..."
        lines.append(f"{role}: {content}")

    return "\n".join(lines) if lines else "None"


# ── Follow-up query rewriting ────────────────────────────────────────────────

_VAGUE_SIGNALS = {"it", "that", "this", "those", "them", "they", "he", "she", "his", "her"}

def _contextualize_query(
    query: str,
    chat_history: Optional[List[Dict[str, str]]],
) -> str:
    """
    Rewrite vague follow-up queries into standalone search queries.

    If the user types "only two" or "tell me more about that", the raw text
    is useless for vector search. This function detects vague follow-ups
    and uses a fast LLM call to rewrite them using conversation context.

    Returns the original query unchanged if it's already specific enough.
    """
    if not chat_history:
        return query

    words = query.lower().split()
    is_short   = len(words) <= 6
    has_pronoun = bool(set(words) & _VAGUE_SIGNALS)

    if not is_short and not has_pronoun:
        return query    # Query is specific enough — skip rewriting

    # Build compact history from last 2 exchanges
    recent = chat_history[-4:]
    history_lines = []
    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        text = msg["content"][:300]
        history_lines.append(f"{role}: {text}")
    history_text = "\n".join(history_lines)

    try:
        rewriter = ChatOpenAI(
            model=LLM_MODEL,
            temperature=0,
            openai_api_key=OPENAI_API_KEY,
            max_tokens=100,
        )
        result = rewriter.invoke(
            "You are a search query rewriter. Given the conversation history "
            "and a follow-up message, rewrite the follow-up as a standalone "
            "search query that captures the full intent. Return ONLY the "
            "rewritten query — no explanation, no quotes.\n\n"
            f"Conversation:\n{history_text}\n\n"
            f"Follow-up: {query}\n\n"
            "Standalone query:"
        )
        rewritten = result.content.strip().strip('"').strip("'")
        return rewritten if rewritten else query
    except Exception:
        return query    # On any failure, use the original query


# ── Context formatting ────────────────────────────────────────────────────────

def format_docs_with_metadata(docs: List[Document]) -> str:
    """
    Format retrieved chunks into a numbered context block for the LLM.

    Each chunk gets a SOURCE N header so the LLM can reference it exactly.
    """
    if not docs:
        return ""

    formatted = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "Unknown")
        page   = doc.metadata.get("page", "Unknown")

        filename     = Path(source).name if ("/" in source or "\\" in source) else source
        page_display = (page + 1) if isinstance(page, int) else page

        formatted.append(
            f"[SOURCE {i}]\n"
            f"Document: {filename}\n"
            f"Page: {page_display}\n"
            f"Content: {doc.page_content.strip()}"
        )

    return "\n\n---\n\n".join(formatted)


# ── Sources list builder ─────────────────────────────────────────────────────

def _build_sources(docs: List[Document], scores: List[float]) -> List[Dict]:
    """Build the sources list from retrieved docs + scores."""
    sources = []
    for doc, score in zip(docs, scores):
        raw_source = doc.metadata.get("source", "Unknown")
        filename   = Path(raw_source).name if ("/" in raw_source or "\\" in raw_source) else raw_source
        page       = doc.metadata.get("page", "Unknown")
        page_disp  = (page + 1) if isinstance(page, int) else page
        sources.append({
            "filename": filename,
            "page":     page_disp,
            "score":    round(score, 4),
            "preview":  doc.page_content[:200],
        })
    return sources


# ── Prepare context (retrieval + guardrail — no LLM call) ───────────────────

def prepare_rag_context(
    vector_store: Chroma,
    query: str,
    filter_docs: Optional[List[str]] = None,
    chat_history: Optional[List[Dict[str, str]]] = None,
    all_docs: Optional[List[Document]] = None,
    top_k_override: Optional[int] = None,
) -> Dict:
    """
    Run retrieval, reranking, and guardrail checks.

    Returns a dict with:
        ready              : bool — True if LLM can generate an answer
        context            : str  — formatted source docs (empty if not ready)
        sources            : list — source metadata for UI
        avg_score          : float — mean retrieval score
        retrieval_time     : float
        chat_history_formatted : str
        fallback_triggered : bool
        guardrail_reason   : str
        retrieval_cfg      : RetrievalConfig
        docs / scores      : raw retrieval results
    """
    query = query.strip()
    if not query:
        return {
            "ready": False,
            "answer": "Please enter a question.",
            "sources": [],
            "fallback_triggered": True,
            "guardrail_reason": "empty_query",
        }

    # ── Cache check (skip when chat history present) ─────────────────────────
    has_history = bool(chat_history)
    if not has_history:
        cached = get_cached_result(query, filter_docs)
        if cached is not None:
            cached["from_cache"] = True
            cached["ready"] = True
            return cached

    # ── Rewrite vague follow-ups into standalone search queries ─────────────
    search_query = _contextualize_query(query, chat_history)
    trace_rewritten = search_query if search_query != query else None

    # ── Query classification ─────────────────────────────────────────────────
    retrieval_cfg = classify_query(search_query)
    if top_k_override and top_k_override > retrieval_cfg.top_k:
        retrieval_cfg.top_k = top_k_override
        retrieval_cfg.fetch_k = max(retrieval_cfg.fetch_k, top_k_override * 3)

    #���─ Retrieve with scores (for guardrail) ─────────────────────────────────
    t_retrieval = time.time()
    docs, scores = retrieve_with_scores(
        vector_store,
        retrieval_cfg.query,
        k=retrieval_cfg.top_k,
        filter_docs=filter_docs,
    )
    retrieval_time = time.time() - t_retrieval

    # Keep pre-rerank state for query trace
    raw_docs, raw_scores = list(docs), list(scores)

    # ── Rerank ───────────────────────────────────────────────────────────────
    # For upload collections (top_k_override set), use a lower threshold
    # and skip per-page dedup so all chunks contribute to the answer.
    rerank_threshold = 0.01 if top_k_override else SIMILARITY_THRESHOLD
    docs, scores = rerank_by_score(
        docs, scores, top_k=retrieval_cfg.top_k,
        threshold=rerank_threshold, deduplicate=not bool(top_k_override),
    )

    # ── Guardrail ────────────────────────────────────────────────────────────
    confidence_ok, reason = check_retrieval_confidence(docs, scores)

    avg_score = round(sum(scores) / len(scores), 4) if scores else 0.0
    sources = _build_sources(docs, scores)
    formatted_history = _format_chat_history(chat_history)

    if not confidence_ok:
        log_query_event(
            query=query, response=FALLBACK_RESPONSE,
            response_time=0.0, retrieval_time=retrieval_time,
            num_chunks=0, source_documents=[],
            error=f"guardrail_triggered:{reason}",
        )
        return {
            "ready":             False,
            "answer":            FALLBACK_RESPONSE,
            "sources":           [],
            "response_time":     0.0,
            "retrieval_time":    retrieval_time,
            "num_chunks":        0,
            "confidence_ok":     False,
            "fallback_triggered": True,
            "guardrail_reason":  reason,
            "token_usage":       0,
            "avg_score":         0.0,
            "trace":             None,
        }

    # ── Format context ───────────────────────────────────────────────────────
    context = format_docs_with_metadata(docs)

    # ── Build query trace for "Under the Hood" panel ─────────────────────────
    trace = {
        "original_query": query,
        "rewritten_query": trace_rewritten,
        "query_classification": {
            "type": retrieval_cfg.query_type.value,
            "top_k": retrieval_cfg.top_k,
            "fetch_k": retrieval_cfg.fetch_k,
        },
        "retrieved_chunks": [
            {"source": d.metadata.get("source", "?"), "page": d.metadata.get("page", "?"),
             "score": round(s, 4), "preview": d.page_content[:100]}
            for d, s in zip(raw_docs, raw_scores)
        ],
        "reranked_chunks": [
            {"source": d.metadata.get("source", "?"), "page": d.metadata.get("page", "?"),
             "score": round(s, 4), "preview": d.page_content[:100]}
            for d, s in zip(docs, scores)
        ],
        "context_preview": context[:500] + "..." if len(context) > 500 else context,
    }

    return {
        "ready":               True,
        "context":             context,
        "sources":             sources,
        "avg_score":           avg_score,
        "retrieval_time":      retrieval_time,
        "num_chunks":          len(docs),
        "chat_history_formatted": formatted_history,
        "retrieval_cfg":       retrieval_cfg,
        "docs":                docs,
        "scores":              scores,
        "fallback_triggered":  False,
        "confidence_ok":       True,
        "all_docs":            all_docs,
        "filter_docs":         filter_docs,
        "trace":               trace,
    }


# ── Streaming response generator ─────────────────────────────────────────────

def stream_rag_response(
    prepared: Dict,
    query: str,
) -> Generator[str, None, None]:
    """
    Stream LLM tokens given pre-retrieved context.

    Yields individual string chunks as they arrive from the model.
    The caller collects them to form the full answer.
    """
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        openai_api_key=OPENAI_API_KEY,
        streaming=True,
    )
    prompt = ChatPromptTemplate.from_template(_load_prompt_template())
    chain  = prompt | llm | StrOutputParser()

    inputs = {
        "context":      prepared["context"],
        "question":     query,
        "chat_history": prepared["chat_history_formatted"],
    }

    for chunk in chain.stream(inputs):
        yield chunk


# ── Non-streaming query (backward-compatible for API) ────────────────────────

def run_query(
    vector_store: Chroma,
    query: str,
    filter_docs: Optional[List[str]] = None,
    chat_history: Optional[List[Dict[str, str]]] = None,
    all_docs: Optional[List[Document]] = None,
    top_k_override: Optional[int] = None,
) -> Dict:
    """
    Run a complete RAG query (non-streaming).

    Used by the FastAPI endpoint and any non-Streamlit caller.
    Streamlit uses prepare_rag_context + stream_rag_response instead.
    """
    prepared = prepare_rag_context(
        vector_store, query, filter_docs, chat_history, all_docs,
        top_k_override=top_k_override,
    )

    if not prepared.get("ready"):
        return prepared

    # Check if this was a cache hit
    if prepared.get("from_cache"):
        return prepared

    # ── Invoke LLM (non-streaming) ───────────────────────────────────────────
    t_response = time.time()
    try:
        llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            openai_api_key=OPENAI_API_KEY,
        )
        prompt = ChatPromptTemplate.from_template(_load_prompt_template())
        chain  = prompt | llm | StrOutputParser()

        answer = chain.invoke({
            "context":      prepared["context"],
            "question":     query,
            "chat_history": prepared["chat_history_formatted"],
        })
        response_time = time.time() - t_response
    except Exception as exc:
        response_time = time.time() - t_response
        log_query_event(
            query=query, response="",
            response_time=response_time,
            retrieval_time=prepared["retrieval_time"],
            num_chunks=prepared["num_chunks"],
            source_documents=[], error=str(exc),
        )
        return {
            "answer": f"Error: {exc}",
            "sources": [],
            "error": str(exc),
        }

    # ── Log and return ───────────────────────────────────────────────────────
    source_names   = list({s["filename"] for s in prepared["sources"]})
    token_estimate = len(query.split()) + len(answer.split())

    log_query_event(
        query=query, response=answer,
        response_time=response_time,
        retrieval_time=prepared["retrieval_time"],
        num_chunks=prepared["num_chunks"],
        source_documents=source_names,
        token_usage=token_estimate,
    )

    result = {
        "answer":            answer,
        "sources":           prepared["sources"],
        "response_time":     response_time,
        "retrieval_time":    prepared["retrieval_time"],
        "num_chunks":        prepared["num_chunks"],
        "confidence_ok":     True,
        "fallback_triggered": False,
        "token_usage":       token_estimate,
        "from_cache":        False,
        "avg_score":         prepared["avg_score"],
        "query_type":        prepared["retrieval_cfg"].query_type.value,
        "context":           prepared.get("context", ""),
    }

    # Cache only standalone queries (no chat history)
    if not chat_history:
        set_cached_result(query, result, filter_docs)

    return result
