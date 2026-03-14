"""
app/streamlit_ui.py
────────────────────
Production Streamlit UI for the Enterprise RAG Chatbot.

Features:
  - Library mode  : query pre-loaded document library with multi-doc selection
  - Upload mode   : upload one or more PDFs and query them instantly
  - Metrics panel : real-time query stats (latency, token usage, top docs)
  - Citation display: grouped by source document
  - Hallucination guardrails: fallback message on low-confidence retrieval
"""
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st

# ── Path bootstrap ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from api.rag_api           import run_query, format_docs_with_metadata
from ingestion.embedding_pipeline import ingest_uploaded_pdf
from monitoring.metrics    import MetricsTracker
from vectorstore.chroma_manager import ChromaManager
from config.settings       import OPENAI_API_KEY, VECTORSTORE_DIR


# ══════════════════════════════════════════════════════════════════════════════
# Page config
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Enterprise RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.metric-card {
    background: #1e293b; border-radius: 8px;
    padding: 12px 16px; margin: 4px 0;
}
.source-chip {
    background: #1e3a5f; border-radius: 4px;
    padding: 2px 8px; font-size: 0.8em;
    color: #93c5fd; display: inline-block; margin: 2px;
}
.fallback-box {
    background: #1c1a12; border-left: 3px solid #f59e0b;
    padding: 12px; border-radius: 4px; color: #fcd34d;
}
.guardrail-badge {
    background: #7f1d1d; color: #fca5a5;
    border-radius: 4px; padding: 2px 8px; font-size: 0.75em;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Session state initialisation
# ══════════════════════════════════════════════════════════════════════════════
def _init_session():
    defaults = {
        "messages":          [],
        "metrics":           MetricsTracker(),
        "library_store":     None,
        "upload_store":      None,
        "upload_filenames":  [],   # currently loaded file names
        "selected_docs":     [],   # user-selected subset for filtering
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_session()


# ══════════════════════════════════════════════════════════════════════════════
# Cached library loader (loaded once; shared across all sessions)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading document library…")
def _load_library() -> Optional[ChromaManager]:
    """Load the persisted ChromaDB collection for the pre-built library."""
    if not OPENAI_API_KEY:
        return None
    try:
        manager = ChromaManager(collection_name="library", persist=True)
        count   = manager.get_chunk_count()
        if count == 0:
            return None
        return manager
    except Exception:
        return None
    
def _render_sources(sources: List[Dict]) -> None:
    """Render a collapsed source-citation panel below an answer."""
    with st.expander(f"📎 Sources ({len(sources)} chunks retrieved)", expanded=False):
        # Group by document
        grouped: Dict[str, List[Dict]] = {}
        for s in sources:
            grouped.setdefault(s["filename"], []).append(s)

        for filename, chunks in grouped.items():
            st.markdown(f"**📄 {filename}**")
            for chunk in chunks:
                st.markdown(
                    f"- Page {chunk['page']} &nbsp; "
                    f"<small style='color:#64748b'>score: {chunk['score']:.3f}</small><br>"
                    f"<small style='color:#94a3b8'>{chunk['preview']}…</small>",
                    unsafe_allow_html=True,
                )



# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🤖 RAG Chatbot")
    st.markdown("---")

    # API key check
    if not OPENAI_API_KEY:
        st.error("❌ OPENAI_API_KEY not set in .env")
        st.stop()

    # Mode selector
    mode = st.radio(
        "**Query Mode**",
        ["📚 Library Documents", "📄 Upload Your PDF"],
        index=0,
    )
    st.markdown("---")

    # ── Library mode controls ──────────────────────────────────────────────
    if mode == "📚 Library Documents":
        lib_manager = _load_library()

        if lib_manager is None:
            st.warning(
                "No library found. Run `python ingest.py` to index your PDFs."
            )
        else:
            doc_list = lib_manager.get_document_list()
            chunk_count = lib_manager.get_chunk_count()
            st.success(f"✅ Library: **{len(doc_list)}** doc(s) · {chunk_count} chunks")

            if doc_list:
                st.markdown("**Filter documents:**")
                selected = st.multiselect(
                    "Select documents to query (empty = all)",
                    options=doc_list,
                    default=[],
                    help="Leave empty to query all documents in the library.",
                )
                st.session_state.selected_docs = selected or doc_list
                st.session_state.library_store = lib_manager

    # ── Upload mode controls ───────────────────────────────────────────────
    else:
        st.markdown("**Upload PDFs**")
        uploaded_files = st.file_uploader(
            "Choose one or more PDF files",
            type=["pdf"],
            accept_multiple_files=True,
        )

        if uploaded_files:
            new_names = sorted(f.name for f in uploaded_files)
            if new_names != sorted(st.session_state.upload_filenames):
                # New set of files — rebuild the in-memory store
                with st.spinner("Indexing uploaded PDFs…"):
                    all_chunks = []
                    # Use a fresh in-memory collection
                    manager = ChromaManager(
                        collection_name="upload_session",
                        persist=False,
                    )
                    for uf in uploaded_files:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                            tmp.write(uf.read())
                            tmp_path = tmp.name
                        from ingestion.pdf_loader import load_pdf, patch_metadata_source
                        from ingestion.chunking   import split_documents
                        docs   = load_pdf(tmp_path)
                        docs   = patch_metadata_source(docs, uf.name)
                        chunks = split_documents(docs)
                        manager.add_documents(chunks)
                        all_chunks.extend(chunks)
                        os.unlink(tmp_path)

                    st.session_state.upload_store     = manager
                    st.session_state.upload_filenames = new_names
                    st.session_state.messages         = []  # Clear chat on new upload

            # Document selector for uploaded files
            store = st.session_state.upload_store
            if store:
                doc_list = store.get_document_list()
                total_chunks = store.get_chunk_count()
                st.success(
                    f"✅ **{len(doc_list)}** file(s) · {total_chunks} chunks indexed"
                )
                for name in doc_list:
                    st.markdown(f"<span class='source-chip'>📄 {name}</span>", unsafe_allow_html=True)

                selected = st.multiselect(
                    "Filter to specific files:",
                    options=doc_list,
                    default=[],
                )
                st.session_state.selected_docs = selected or doc_list

    st.markdown("---")

    # ── Clear chat ─────────────────────────────────────────────────────────
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()

    # ── Metrics dashboard ──────────────────────────────────────────────────
    with st.expander("📊 Session Metrics", expanded=False):
        m = st.session_state.metrics.to_dict()
        col1, col2 = st.columns(2)
        col1.metric("Total Queries",    m["total_queries"])
        col2.metric("Success Rate",     f"{m['success_rate_pct']}%")
        col1.metric("Avg Latency",      f"{m['avg_response_time_ms']} ms")
        col2.metric("Avg Retrieval",    f"{m['avg_retrieval_time_ms']} ms")
        col1.metric("P95 Latency",      f"{m['p95_response_time_ms']} ms")
        col2.metric("Tokens Used",      m["total_tokens_used"])
        col1.metric("Fallbacks",        m["fallback_responses"])
        col2.metric("Failed",           m["failed_queries"])

        if m["top_documents"]:
            st.markdown("**Most queried documents:**")
            for doc_name, count in m["top_documents"]:
                st.markdown(f"- `{doc_name}` — {count} quer{'y' if count == 1 else 'ies'}")


# ══════════════════════════════════════════════════════════════════════════════
# Main chat interface
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("# 🤖 Enterprise RAG Chatbot")
st.caption(
    "Library mode: query pre-loaded documents. "
    "Upload mode: upload any PDF and ask questions instantly."
)

# Determine active vector store
active_store: Optional[ChromaManager] = None
if mode == "📚 Library Documents":
    active_store = st.session_state.get("library_store")
else:
    active_store = st.session_state.get("upload_store")

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            _render_sources(msg["sources"])   # defined below



# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input(
    "Ask a question about your documents…",
    disabled=(active_store is None),
):
    if active_store is None:
        st.warning("No documents loaded yet. Please index your library or upload a PDF.")
        st.stop()

    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Run query
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            result = run_query(
                vector_store=active_store.load(),
                query=prompt,
                filter_docs=st.session_state.selected_docs or None,
            )

        answer  = result["answer"]
        sources = result.get("sources", [])

        # Render answer
        if result.get("fallback_triggered"):
            st.markdown(
                f"<div class='fallback-box'>{answer}<br>"
                f"<span class='guardrail-badge'>⚠️ guardrail triggered — "
                f"{result.get('guardrail_reason', '')}</span></div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(answer)
            if sources:
                _render_sources(sources)

        # Latency pill
        if result.get("response_time"):
            st.caption(
                f"⏱ {result['response_time']*1000:.0f} ms response · "
                f"{result['retrieval_time']*1000:.0f} ms retrieval · "
                f"{result.get('num_chunks', 0)} chunks"
            )

    # Update session state
    st.session_state.messages.append({
        "role":    "assistant",
        "content": answer,
        "sources": sources,
    })
    st.session_state.metrics.record_query(
        response_time=result.get("response_time",  0.0),
        retrieval_time=result.get("retrieval_time", 0.0),
        source_documents=[s["filename"] for s in sources],
        token_usage=result.get("token_usage", 0),
        fallback=result.get("fallback_triggered", False),
    )


# ── Empty state ───────────────────────────────────────────────────────────────
if active_store is None:
    st.info(
        "👈 **Get started:**\n\n"
        "- **Library mode** — run `python ingest.py` to index your PDFs, then ask questions.\n"
        "- **Upload mode** — drag and drop any PDF in the sidebar."
    )
