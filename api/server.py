"""
api/server.py
──────────────
FastAPI server for the RAG chatbot.

Wraps the existing run_query() function — no RAG logic lives here.

Endpoints:
    GET  /health       → liveness check
    GET  /documents    → list documents in library
    POST /query        → run a RAG query, returns structured JSON
    POST /upload       → upload and process files for querying

Cost controls (applied to /query):
    1. Rate limiting  — 20 requests/minute per IP (via slowapi + Redis)
    2. Input length   — query capped at MAX_QUERY_LENGTH characters
    3. Token budget   — query capped at MAX_INPUT_TOKENS tokens (tiktoken)

Run:
    uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
"""
import shutil
import sys
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

import tiktoken
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from api.rag_api import run_query
from api.rate_limiter import limiter
from config.settings import (
    MAX_INPUT_TOKENS,
    MAX_QUERY_LENGTH,
    RATE_LIMIT,
)
from vectorstore.chroma_manager import ChromaManager


# ── Request / Response models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    filter_docs: Optional[List[str]] = None
    upload_id: Optional[str] = None


# ── Startup: load vector store once, reuse across all requests ────────────────

_manager: Optional[ChromaManager] = None
_upload_stores: Dict[str, dict] = {}
_tokenizer = tiktoken.get_encoding("cl100k_base")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _manager
    _manager = ChromaManager(collection_name="library", persist=True)
    yield


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="RAG Chatbot API", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/documents")
def list_documents():
    """List unique document filenames in the library collection."""
    try:
        docs = _manager.get_document_list()
        filenames = [
            Path(d).name if ("/" in d or "\\" in d) else d
            for d in docs
        ]
        return {"documents": sorted(set(filenames))}
    except Exception:
        return {"documents": []}


@app.post("/upload")
async def upload_files(
    files: List[UploadFile] = File(...),
    upload_id: Optional[str] = None,
):
    """Upload, process, and index documents for querying.

    If upload_id is provided and exists, new files are added to the
    existing collection so users can accumulate multiple documents.
    """
    is_existing = upload_id is not None and upload_id in _upload_stores
    if not is_existing:
        upload_id = uuid.uuid4().hex[:8]

    tmp_dir = tempfile.mkdtemp()

    try:
        saved_paths: List[str] = []
        for f in files:
            dest = Path(tmp_dir) / f.filename
            with open(dest, "wb") as out:
                content = await f.read()
                out.write(content)
            saved_paths.append(str(dest))

        from ingestion.chunking import split_documents
        from ingestion.pdf_loader import load_file

        all_docs = []
        for fpath in saved_paths:
            all_docs.extend(load_file(fpath))

        if not all_docs:
            return JSONResponse(
                status_code=400,
                content={"error": "No content could be extracted from the uploaded files."},
            )

        chunks = split_documents(all_docs)

        if is_existing:
            manager = _upload_stores[upload_id]["manager"]
            manager.add_documents(chunks)
            _upload_stores[upload_id]["filenames"].extend(
                Path(p).name for p in saved_paths
            )
            _upload_stores[upload_id]["chunk_count"] += len(chunks)
        else:
            manager = ChromaManager(
                collection_name=f"upload_{upload_id}",
                persist=False,
            )
            manager.add_documents(chunks)
            _upload_stores[upload_id] = {
                "manager": manager,
                "filenames": [Path(p).name for p in saved_paths],
                "chunk_count": len(chunks),
            }

        store = _upload_stores[upload_id]
        return {
            "upload_id": upload_id,
            "filenames": store["filenames"],
            "chunk_count": store["chunk_count"],
            "page_count": len(all_docs),
        }
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.post("/query")
@limiter.limit(RATE_LIMIT)
def query(request: Request, req: QueryRequest):
    # Guard 1: input length
    if len(req.query) > MAX_QUERY_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Query too long. Maximum {MAX_QUERY_LENGTH} characters allowed.",
        )

    # Guard 2: token budget
    token_count = len(_tokenizer.encode(req.query))
    if token_count > MAX_INPUT_TOKENS:
        raise HTTPException(
            status_code=400,
            detail=f"Query exceeds token limit. Maximum {MAX_INPUT_TOKENS} tokens allowed.",
        )

    # Pick vector store: uploaded docs or library
    top_k_override = None
    if req.upload_id and req.upload_id in _upload_stores:
        store_info = _upload_stores[req.upload_id]
        vector_store = store_info["manager"].load()
        # For small uploaded collections, retrieve more chunks so broad
        # questions like "What is this document about?" get full context.
        chunk_count = store_info["chunk_count"]
        if chunk_count <= 30:
            top_k_override = min(chunk_count, 15)
    else:
        vector_store = _manager.load()

    result = run_query(
        vector_store=vector_store,
        query=req.query,
        filter_docs=req.filter_docs,
        top_k_override=top_k_override,
    )
    return result
