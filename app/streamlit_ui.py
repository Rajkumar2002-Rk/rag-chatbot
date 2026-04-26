"""
app/streamlit_ui.py
────────────────────
Polished demo UI for the RAG Chatbot portfolio project.
"""
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from api.rag_api import (
    prepare_rag_context,
    stream_rag_response,
    run_query,
    format_docs_with_metadata,
)
from monitoring.metrics    import MetricsTracker
from monitoring.feedback   import save_feedback
from vectorstore.chroma_manager import ChromaManager
from config.settings       import OPENAI_API_KEY


# ══════════════════════════════════════════════════════════════════════════════
# Page config
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# CSS — clean, modern, professional
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Global ─────────────────────────────────────────────────────────── */
.block-container { max-width: 900px; }

/* ── Hero / Welcome ─────────────────────────────────────────────────── */
.hero-title {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #f472b6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0;
    line-height: 1.2;
}
.hero-subtitle {
    color: #94a3b8;
    font-size: 1.05rem;
    margin-top: 4px;
    margin-bottom: 24px;
}

/* ── Step cards ─────────────────────────────────────────────────────── */
.step-card {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 20px;
    height: 100%;
    transition: border-color 0.2s;
}
.step-card:hover { border-color: #3b82f6; }
.step-number {
    display: inline-block;
    background: linear-gradient(135deg, #3b82f6, #8b5cf6);
    color: white;
    width: 28px; height: 28px;
    border-radius: 50%;
    text-align: center;
    line-height: 28px;
    font-size: 0.85rem;
    font-weight: 700;
    margin-bottom: 8px;
}
.step-title {
    font-size: 1rem;
    font-weight: 600;
    color: #e2e8f0;
    margin-bottom: 4px;
}
.step-desc {
    color: #94a3b8;
    font-size: 0.85rem;
    line-height: 1.4;
}

/* ── File chips ─────────────────────────────────────────────────────── */
.file-chip {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 4px 12px;
    font-size: 0.82rem;
    color: #93c5fd;
    display: inline-block;
    margin: 2px 4px 2px 0;
}

/* ── Source citations ───────────────────────────────────────────────── */
.source-item {
    background: #0f172a;
    border-left: 3px solid #3b82f6;
    border-radius: 0 8px 8px 0;
    padding: 8px 14px;
    margin: 6px 0;
    font-size: 0.85rem;
}
.source-doc { color: #93c5fd; font-weight: 600; }
.source-page { color: #64748b; font-size: 0.8rem; }
.source-preview { color: #94a3b8; font-size: 0.8rem; line-height: 1.3; }

/* ── Fallback message ───────────────────────────────────────────────── */
.fallback-box {
    background: linear-gradient(135deg, #1c1a12 0%, #1a1510 100%);
    border-left: 3px solid #f59e0b;
    padding: 16px;
    border-radius: 0 10px 10px 0;
    color: #fcd34d;
}

/* ── Confidence (only shown in tech mode) ───────────────────────────── */
.confidence-high {
    background: #064e3b; color: #6ee7b7;
    border-radius: 6px; padding: 3px 10px; font-size: 0.75rem;
}
.confidence-medium {
    background: #78350f; color: #fcd34d;
    border-radius: 6px; padding: 3px 10px; font-size: 0.75rem;
}
.confidence-low {
    background: #7c2d12; color: #fdba74;
    border-radius: 6px; padding: 3px 10px; font-size: 0.75rem;
}

/* ── Sidebar branding ───────────────────────────────────────────────── */
.sidebar-brand {
    font-size: 1.3rem;
    font-weight: 700;
    background: linear-gradient(135deg, #60a5fa, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 2px;
}
.sidebar-tagline {
    color: #64748b;
    font-size: 0.8rem;
    margin-bottom: 12px;
}

/* ── Tech details section ───────────────────────────────────────────── */
.tech-label {
    color: #64748b;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 2px;
}
.tech-value {
    color: #94a3b8;
    font-size: 0.85rem;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Session state
# ══════════════════════════════════════════════════════════════════════════════
def _init_session():
    defaults = {
        "messages":          [],
        "metrics":           MetricsTracker(),
        "library_store":     None,
        "upload_store":      None,
        "upload_filenames":  [],
        "upload_chunks":     [],
        "selected_docs":     [],
        "pending_question":  None,
        "show_tech":         False,
        "feedback_given":    {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_session()


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading document library...")
def _load_library() -> Optional[ChromaManager]:
    if not OPENAI_API_KEY:
        return None
    try:
        manager = ChromaManager(collection_name="library", persist=True)
        if manager.get_chunk_count() == 0:
            return None
        return manager
    except Exception:
        return None


@st.cache_resource(show_spinner="Preparing search index...")
def _load_library_docs(_manager) -> List:
    try:
        return _manager.get_all_documents()
    except Exception:
        return []


def _render_sources(sources: List[Dict], show_scores: bool = False) -> None:
    """Clean source citation panel."""
    with st.expander(f"View Sources ({len(sources)})", expanded=False):
        grouped: Dict[str, List[Dict]] = {}
        for s in sources:
            grouped.setdefault(s["filename"], []).append(s)

        for filename, chunks in grouped.items():
            for chunk in chunks:
                score_html = ""
                if show_scores:
                    score_html = f" &middot; <span style='color:#64748b'>{chunk['score']:.3f}</span>"
                st.markdown(
                    f"<div class='source-item'>"
                    f"<span class='source-doc'>{filename}</span>"
                    f" <span class='source-page'>Page {chunk['page']}{score_html}</span><br>"
                    f"<span class='source-preview'>{chunk['preview'][:150]}...</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )


def _render_confidence(avg_score: float) -> None:
    if avg_score >= 0.5:
        css, label = "confidence-high", "High"
    elif avg_score >= 0.3:
        css, label = "confidence-medium", "Medium"
    else:
        css, label = "confidence-low", "Low"
    st.markdown(
        f"<span class='{css}'>Confidence: {label} ({avg_score:.2f})</span>",
        unsafe_allow_html=True,
    )


def _render_trace(trace: dict) -> None:
    """Render the full query trace pipeline (Under the Hood mode only)."""
    with st.expander("Query Trace", expanded=False):
        st.markdown("**Original Query**")
        st.code(trace["original_query"], language=None)

        if trace.get("rewritten_query"):
            st.markdown("**Rewritten Query** (follow-up contextualized)")
            st.code(trace["rewritten_query"], language=None)

        cls = trace["query_classification"]
        st.markdown(f"**Classification:** `{cls['type']}` — top_k={cls['top_k']}, fetch_k={cls['fetch_k']}")

        st.markdown(f"**Retrieved Chunks** ({len(trace['retrieved_chunks'])} pre-rerank)")
        for c in trace["retrieved_chunks"]:
            st.caption(f"`{c['source']}` p.{c['page']} — score: {c['score']}")

        st.markdown(f"**After Reranking** ({len(trace['reranked_chunks'])} chunks)")
        for c in trace["reranked_chunks"]:
            st.caption(f"`{c['source']}` p.{c['page']} — score: {c['score']}")

        st.markdown("**Context Sent to LLM**")
        st.code(trace["context_preview"], language=None)


_FEEDBACK_ICONS = {"positive": "👍", "neutral": "😐", "negative": "👎"}


def _render_feedback(msg_index: int, messages: list) -> None:
    """Render 3-button feedback (thumbs down / neutral / thumbs up)."""
    fb_key = str(msg_index)
    if fb_key in st.session_state.feedback_given:
        given = st.session_state.feedback_given[fb_key]
        icon = _FEEDBACK_ICONS.get(given, "")
        st.caption(f"{icon} Thanks for the feedback")
        return

    def _save(fb_type):
        user_query = ""
        if msg_index > 0 and messages[msg_index - 1]["role"] == "user":
            user_query = messages[msg_index - 1]["content"]
        save_feedback(
            query=user_query,
            answer=messages[msg_index]["content"],
            feedback=fb_type,
            sources=messages[msg_index].get("sources", []),
        )
        st.session_state.feedback_given[fb_key] = fb_type

    cols = st.columns([1, 1, 1, 8])
    with cols[0]:
        if st.button("👍", key=f"fb_up_{msg_index}", help="Good answer"):
            _save("positive")
            st.rerun()
    with cols[1]:
        if st.button("😐", key=f"fb_neu_{msg_index}", help="Okay answer"):
            _save("neutral")
            st.rerun()
    with cols[2]:
        if st.button("👎", key=f"fb_down_{msg_index}", help="Bad answer"):
            _save("negative")
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        "<div class='sidebar-brand'>AI Document Assistant</div>"
        "<div class='sidebar-tagline'>Ask questions, get answers from your documents</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    if not OPENAI_API_KEY:
        st.error("OPENAI_API_KEY not configured")
        st.stop()

    # ── Mode selector ─────────────────────────────────────────────────────
    mode = st.radio(
        "**Choose a mode**",
        ["Sample Documents", "Upload Your Own"],
        index=0,
        help="Sample Documents: pre-loaded library. Upload: your own files.",
    )
    st.markdown("---")

    # ── Library mode ──────────────────────────────────────────────────────
    if mode == "Sample Documents":
        lib_manager = _load_library()

        if lib_manager is None:
            st.warning("No library found. Run `python ingest.py` first.")
        else:
            doc_list    = lib_manager.get_document_list()
            chunk_count = lib_manager.get_chunk_count()
            st.success(f"**{len(doc_list)}** documents ready")

            if doc_list:
                selected = st.multiselect(
                    "Filter documents:",
                    options=doc_list,
                    default=[],
                    help="Leave empty to search all documents.",
                )
                st.session_state.selected_docs = selected or doc_list
                st.session_state.library_store = lib_manager

    # ── Upload mode ───────────────────────────────────────────────────────
    else:
        uploaded_files = st.file_uploader(
            "Drop files here",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            help="Supports PDF, Word, and text files",
        )

        if uploaded_files:
            new_names = sorted(f.name for f in uploaded_files)
            if new_names != sorted(st.session_state.upload_filenames):
                with st.spinner("Processing documents..."):
                    all_chunks = []
                    manager = ChromaManager(
                        collection_name="upload_session",
                        persist=False,
                    )
                    from ingestion.pdf_loader import load_file, patch_metadata_source
                    from ingestion.chunking   import split_documents
                    for uf in uploaded_files:
                        tmp_path = None
                        try:
                            suffix = Path(uf.name).suffix or ".pdf"
                            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                                tmp.write(uf.read())
                                tmp_path = tmp.name
                            docs   = load_file(tmp_path)
                            docs   = patch_metadata_source(docs, uf.name)
                            chunks = split_documents(docs)
                            manager.add_documents(chunks)
                            all_chunks.extend(chunks)
                        except Exception as exc:
                            st.error(f"Could not process **{uf.name}**: {exc}")
                        finally:
                            if tmp_path and os.path.exists(tmp_path):
                                os.unlink(tmp_path)

                    if all_chunks:
                        st.session_state.upload_store     = manager
                        st.session_state.upload_filenames = new_names
                        st.session_state.upload_chunks    = all_chunks
                        st.session_state.messages         = []
                    else:
                        st.error("No content could be extracted from the files.")

            store = st.session_state.upload_store
            if store and not store.is_healthy:
                st.session_state.upload_store    = None
                st.session_state.upload_filenames = []
                st.session_state.upload_chunks   = []
                store = None
                st.info("Session expired. Please re-upload your files.")
            if store:
                doc_list = store.get_document_list()
                st.success(f"**{len(doc_list)}** file(s) ready")
                for name in doc_list:
                    st.markdown(f"<span class='file-chip'>{name}</span>", unsafe_allow_html=True)

                if len(doc_list) > 1:
                    selected = st.multiselect(
                        "Filter files:", options=doc_list, default=[],
                    )
                    st.session_state.selected_docs = selected or doc_list
                else:
                    st.session_state.selected_docs = doc_list

    st.markdown("---")

    # ── Example questions ─────────────────────────────────────────────────
    if mode == "Sample Documents":
        st.markdown("**Try these questions:**")
        example_questions = [
            "What programming languages does Raj know?",
            "What ML projects has Raj built?",
            "How does the Transformer use attention?",
            "What exams did GPT-4 pass?",
        ]
        for q in example_questions:
            if st.button(q, key=f"eq_{q}", use_container_width=True):
                st.session_state.pending_question = q
                st.rerun()
        st.markdown("---")

    # ── Clear + Tech toggle ───────────────────────────────────────────────
    col_clear, col_tech = st.columns(2)
    with col_clear:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col_tech:
        st.session_state.show_tech = st.toggle("Under the Hood", value=st.session_state.show_tech)

    # ── Technical details (only when toggled) ─────────────────────────────
    if st.session_state.show_tech:
        st.markdown("---")
        st.markdown("**System Stack**")
        tech_items = [
            ("Language Model", "GPT-4o-mini"),
            ("Embeddings", "text-embedding-3-small"),
            ("Vector DB", "ChromaDB (cosine)"),
            ("Search", "Hybrid (BM25 + MMR)"),
            ("Framework", "LangChain"),
            ("Deployment", "Docker / AWS EC2"),
        ]
        for label, value in tech_items:
            st.markdown(
                f"<div><span class='tech-label'>{label}</span><br>"
                f"<span class='tech-value'>{value}</span></div>",
                unsafe_allow_html=True,
            )
        st.markdown("")

        with st.expander("Session Metrics"):
            m = st.session_state.metrics.to_dict()
            c1, c2 = st.columns(2)
            c1.metric("Queries",      m["total_queries"])
            c2.metric("Success",      f"{m['success_rate_pct']}%")
            c1.metric("Avg Latency",  f"{m['avg_response_time_ms']}ms")
            c2.metric("Cache Hits",   m["cache_hits"])

        with st.expander("User Feedback"):
            from monitoring.feedback import get_feedback_stats
            fb = get_feedback_stats()
            f1, f2, f3 = st.columns(3)
            f1.metric("👍", fb["positive"])
            f2.metric("😐", fb["neutral"])
            f3.metric("👎", fb["negative"])


# ══════════════════════════════════════════════════════════════════════════════
# Main area
# ══════════════════════════════════════════════════════════════════════════════
show_tech = st.session_state.show_tech

# ── Determine active store ───────────────────────────────────────────────────
active_store: Optional[ChromaManager] = None
if mode == "Sample Documents":
    active_store = st.session_state.get("library_store")
else:
    active_store = st.session_state.get("upload_store")

# ── Welcome screen (only when chat is empty) ─────────────────────────────────
if not st.session_state.messages:
    st.markdown(
        "<div class='hero-title'>AI Document Assistant</div>"
        "<div class='hero-subtitle'>"
        "Upload any document and ask questions — get accurate, cited answers instantly."
        "</div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        st.markdown(
            "<div class='step-card'>"
            "<div class='step-number'>1</div>"
            "<div class='step-title'>Choose Documents</div>"
            "<div class='step-desc'>"
            "Use the sample library or upload your own PDF, Word, or text files."
            "</div></div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            "<div class='step-card'>"
            "<div class='step-number'>2</div>"
            "<div class='step-title'>Ask a Question</div>"
            "<div class='step-desc'>"
            "Type any question in the chat box below. Try the example questions in the sidebar."
            "</div></div>",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            "<div class='step-card'>"
            "<div class='step-number'>3</div>"
            "<div class='step-title'>Get Cited Answers</div>"
            "<div class='step-desc'>"
            "Every answer includes the exact source document and page number."
            "</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("")

    # Architecture (only in tech mode)
    if show_tech:
        st.caption(
            "**Pipeline:** Document Ingestion → Adaptive Chunking → "
            "OpenAI Embeddings → ChromaDB → Hybrid Search (BM25 + MMR) → "
            "GPT-4o-mini → Cited Answer"
        )

    st.markdown("---")

# ── Chat history ─────────────────────────────────────────────────────────────
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            _render_sources(msg["sources"], show_scores=show_tech)
        if show_tech and msg.get("trace"):
            _render_trace(msg["trace"])
        if msg["role"] == "assistant" and msg.get("content"):
            _render_feedback(i, st.session_state.messages)

# ── Chat input ───────────────────────────────────────────────────────────────
prompt: Optional[str] = None
chat_input = st.chat_input(
    "Ask a question about your documents...",
    disabled=(active_store is None),
)

pending = st.session_state.get("pending_question")
if pending:
    st.session_state.pending_question = None
    prompt = pending
else:
    prompt = chat_input

# ── Handle query ─────────────────────────────────────────────────────────────
if prompt:
    if active_store is None:
        st.warning("Please load documents first using the sidebar.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Resolve all_docs for hybrid search
    all_docs = None
    if mode == "Sample Documents":
        lib = st.session_state.get("library_store")
        if lib:
            all_docs = _load_library_docs(lib)
    else:
        all_docs = st.session_state.get("upload_chunks", [])

    with st.chat_message("assistant"):
        import time as _time
        t_start = _time.time()

        with st.spinner("Searching documents..."):
            prepared = prepare_rag_context(
                vector_store=active_store.load(),
                query=prompt,
                filter_docs=st.session_state.selected_docs or None,
                chat_history=st.session_state.messages,
                all_docs=all_docs,
            )

        sources = prepared.get("sources", [])

        # ── Cache hit ────────────────────────────────────────────────────
        if prepared.get("from_cache"):
            answer = prepared["answer"]
            st.markdown(answer)
            if sources:
                _render_sources(sources, show_scores=show_tech)

        # ── Fallback ─────────────────────────────────────────────────────
        elif not prepared.get("ready"):
            answer = prepared.get("answer", "")
            tip = (
                "Try asking something more specific about the document content."
                if mode != "Sample Documents"
                else "Try one of the example questions in the sidebar."
            )
            st.markdown(
                f"<div class='fallback-box'>"
                f"I couldn't find an answer to that in the documents.<br>"
                f"<small>{tip}</small></div>",
                unsafe_allow_html=True,
            )
            # Show technical reason only in tech mode
            if show_tech:
                st.caption(f"Reason: {prepared.get('guardrail_reason', 'unknown')}")

        # ── Stream response ──────────────────────────────────────────────
        else:
            try:
                answer = st.write_stream(stream_rag_response(prepared, prompt))
            except Exception as exc:
                answer = f"Something went wrong: {exc}"
                st.error(answer)

            if sources:
                _render_sources(sources, show_scores=show_tech)

            # Trace + confidence in tech mode
            if show_tech:
                if prepared.get("trace"):
                    _render_trace(prepared["trace"])
                avg_score = prepared.get("avg_score", 0.0)
                if avg_score > 0:
                    _render_confidence(avg_score)

        # ── Tech-mode timing ─────────────────────────────────────────────
        total_ms     = (_time.time() - t_start) * 1000
        retrieval_ms = prepared.get("retrieval_time", 0.0) * 1000
        num_chunks   = prepared.get("num_chunks", 0)
        if show_tech and retrieval_ms > 0 and not prepared.get("from_cache"):
            st.caption(
                f"{total_ms:.0f}ms total | "
                f"{retrieval_ms:.0f}ms retrieval | "
                f"{num_chunks} chunks"
            )

        # ── Append message to state BEFORE rendering feedback ────────────
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "trace": prepared.get("trace"),
        })

        # ── Show feedback immediately after the answer ───────────────────
        new_msg_index = len(st.session_state.messages) - 1
        _render_feedback(new_msg_index, st.session_state.messages)
    st.session_state.metrics.record_query(
        response_time=(total_ms / 1000),
        retrieval_time=prepared.get("retrieval_time", 0.0),
        source_documents=[s["filename"] for s in sources],
        token_usage=0,
        fallback=prepared.get("fallback_triggered", False),
    )
    st.session_state.metrics.record_cache_event(hit=prepared.get("from_cache", False))


# ── Empty state ──────────────────────────────────────────────────────────────
if active_store is None and mode == "Sample Documents":
    st.info("No documents loaded yet. Run `python ingest.py` to set up the library.")
