"""
api/rag_api.py
───────────────
Core RAG logic — completely decoupled from Streamlit.

All business logic lives here. The UI layer (app/streamlit_ui.py) calls
these functions and renders the results. This separation makes the system
testable, reusable, and easy to wrap with FastAPI if needed.

Public API:
    format_docs_with_metadata(docs)     → formatted context string
    build_rag_chain(retriever)          → LCEL chain
    run_query(vector_store, query, ...) → structured result dict
"""
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain.schema import Document
from langchain_chroma import Chroma

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import OPENAI_API_KEY, LLM_MODEL, LLM_TEMPERATURE, PROMPTS_DIR
from retrieval.retriever import (
    FALLBACK_RESPONSE,
    check_retrieval_confidence,
    get_mmr_retriever,
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
    # Inline fallback (should never be needed if Dockerfile is correct)
    return (
        "You are a precise document assistant.\n\n"
        "STRICT RULES:\n"
        "1. Answer ONLY using the SOURCE DOCUMENTS below.\n"
        "2. For EVERY fact, cite as [SOURCE N: filename.pdf | Page: X].\n"
        "3. If not found, say exactly: \"I could not find relevant information in the uploaded document.\"\n\n"
        "SOURCE DOCUMENTS:\n{context}\n\nQuestion: {question}\n\nANSWER (citations mandatory):"
    )


# ── Context formatting ────────────────────────────────────────────────────────

def format_docs_with_metadata(docs: List[Document]) -> str:
    """
    Format retrieved chunks into a numbered context block for the LLM.

    Each chunk gets a SOURCE N header so the LLM can reference it exactly:

        [SOURCE 1]
        Document: china-wikipedia.pdf
        Page: 3
        Content: China, officially the People's Republic of China...

    WHY: A common RAG mistake is stripping metadata before passing context to
    the LLM. Without the SOURCE labels and filenames, the LLM cannot produce
    accurate citations — it either fabricates them or omits them entirely.
    """
    if not docs:
        return ""

    formatted = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "Unknown")
        page   = doc.metadata.get("page", "Unknown")

        # Normalize: strip directory paths to just the filename
        filename     = Path(source).name if ("/" in source or "\\" in source) else source
        page_display = (page + 1) if isinstance(page, int) else page

        formatted.append(
            f"[SOURCE {i}]\n"
            f"Document: {filename}\n"
            f"Page: {page_display}\n"
            f"Content: {doc.page_content.strip()}"
        )

    return "\n\n---\n\n".join(formatted)


# ── Chain builder ─────────────────────────────────────────────────────────────

def build_rag_chain(retriever: Any) -> Any:
    """
    Build the full LCEL RAG chain:
        retriever → format_docs_with_metadata → prompt → LLM → StrOutputParser

    Returns a runnable chain that accepts a question string.
    """
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        openai_api_key=OPENAI_API_KEY,
    )
    prompt = ChatPromptTemplate.from_template(_load_prompt_template())

    chain = (
        {
            "context":  retriever | RunnableLambda(format_docs_with_metadata),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


# ── Main query function ───────────────────────────────────────────────────────

def run_query(
    vector_store: Chroma,
    query: str,
    filter_docs: Optional[List[str]] = None,
) -> Dict:
    """
    Run a complete RAG query with guardrails, timing, and structured logging.

    Pipeline:
      1. Retrieve with similarity scores
      2. Rerank by score (filter low-confidence chunks)
      3. Hallucination guardrail check
      4. Build LCEL chain and invoke LLM
      5. Log the event
      6. Return structured result

    Args:
        vector_store: Chroma store to retrieve from.
        query:        User's question.
        filter_docs:  Optional list of filenames to restrict retrieval to.

    Returns:
        Dict with keys:
            answer, sources, response_time, retrieval_time,
            num_chunks, confidence_ok, fallback_triggered,
            token_usage, error (optional)
    """
    query = query.strip()
    if not query:
        return {"answer": "Please enter a question.", "sources": [], "error": "empty_query"}

    # ── Cache check: skip retrieval + LLM entirely on hit ─────────────────────
    cached = get_cached_result(query, filter_docs)
    if cached is not None:
        cached["from_cache"] = True
        return cached

    # ── Query classification ──────────────────────────────────────────────────
    retrieval_cfg = classify_query(query)

    # ── Step 1: Retrieve with scores ──────────────────────────────────────────
    t_retrieval = time.time()
    docs, scores = retrieve_with_scores(
        vector_store,
        retrieval_cfg.query,          # expanded if ambiguous
        k=retrieval_cfg.top_k,
        filter_docs=filter_docs,
    )
    retrieval_time = time.time() - t_retrieval

    # ── Step 2: Rerank ────────────────────────────────────────────────────────
    docs, scores = rerank_by_score(docs, scores, top_k=retrieval_cfg.top_k)

    # ── Step 3: Hallucination guardrail ───────────────────────────────────────
    confidence_ok, reason = check_retrieval_confidence(docs, scores)

    if not confidence_ok:
        log_query_event(
            query=query,
            response=FALLBACK_RESPONSE,
            response_time=0.0,
            retrieval_time=retrieval_time,
            num_chunks=0,
            source_documents=[],
            error=f"guardrail_triggered:{reason}",
        )
        return {
            "answer":            FALLBACK_RESPONSE,
            "sources":           [],
            "response_time":     0.0,
            "retrieval_time":    retrieval_time,
            "num_chunks":        0,
            "confidence_ok":     False,
            "fallback_triggered": True,
            "guardrail_reason":  reason,
            "token_usage":       0,
        }

    # ── Step 4: Build chain and invoke LLM ───────────────────────────────────
    t_response = time.time()
    try:
        retriever = get_mmr_retriever(
            vector_store,
            filter_docs=filter_docs,
            k=retrieval_cfg.top_k,
            fetch_k=retrieval_cfg.fetch_k,
            lambda_mult=retrieval_cfg.lambda_mult,
        )
        chain     = build_rag_chain(retriever)
        answer    = chain.invoke(query)
        response_time = time.time() - t_response
    except Exception as exc:
        response_time = time.time() - t_response
        log_query_event(
            query=query, response="",
            response_time=response_time, retrieval_time=retrieval_time,
            num_chunks=len(docs), source_documents=[], error=str(exc),
        )
        return {
            "answer":  f"⚠️ Error: {exc}",
            "sources": [],
            "error":   str(exc),
        }

    # ── Step 5: Build sources list ────────────────────────────────────────────
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

    # ── Step 6: Log and return ────────────────────────────────────────────────
    source_names  = list({s["filename"] for s in sources})
    token_estimate = len(query.split()) + len(answer.split())

    log_query_event(
        query=query, response=answer,
        response_time=response_time, retrieval_time=retrieval_time,
        num_chunks=len(docs), source_documents=source_names,
        token_usage=token_estimate,
    )

    result = {
        "answer":            answer,
        "sources":           sources,
        "response_time":     response_time,
        "retrieval_time":    retrieval_time,
        "num_chunks":        len(docs),
        "confidence_ok":     True,
        "fallback_triggered": False,
        "token_usage":       token_estimate,
        "from_cache":        False,
        "query_type":        retrieval_cfg.query_type.value,
    }
    set_cached_result(query, result, filter_docs)
    return result
