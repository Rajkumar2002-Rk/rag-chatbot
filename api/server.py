"""
api/server.py
──────────────
FastAPI server for the RAG chatbot.

Wraps the existing run_query() function — no RAG logic lives here.

Endpoints:
    GET  /health   → liveness check
    POST /query    → run a RAG query, returns structured JSON

Cost controls (applied to /query):
    1. Rate limiting  — 20 requests/minute per IP (via slowapi + Redis)
    2. Input length   — query capped at MAX_QUERY_LENGTH characters
    3. Token budget   — query capped at MAX_INPUT_TOKENS tokens (tiktoken)

Run:
    uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
"""
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import tiktoken
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from api.rag_api import run_query
from api.rate_limiter import limiter
from config.settings import MAX_QUERY_LENGTH, MAX_INPUT_TOKENS, RATE_LIMIT
from vectorstore.chroma_manager import ChromaManager


# ── Request / Response models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    filter_docs: Optional[List[str]] = None


# ── Startup: load vector store once, reuse across all requests ────────────────

_manager: Optional[ChromaManager] = None
_tokenizer = tiktoken.get_encoding("cl100k_base")  # same encoding GPT-3.5/4 use


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


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


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

    result = run_query(
        vector_store=_manager.load(),
        query=req.query,
        filter_docs=req.filter_docs,
    )
    return result
