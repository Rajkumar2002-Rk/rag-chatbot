"""
api/server.py
──────────────
FastAPI server for the RAG chatbot.

Wraps the existing run_query() function — no RAG logic lives here.

Endpoints:
    GET  /health   → liveness check
    POST /query    → run a RAG query, returns structured JSON

Run:
    uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
"""
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from api.rag_api import run_query
from vectorstore.chroma_manager import ChromaManager


# ── Request / Response models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    filter_docs: Optional[List[str]] = None


# ── Startup: load vector store once, reuse across all requests ────────────────

_manager: Optional[ChromaManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _manager
    _manager = ChromaManager(collection_name="library", persist=True)
    yield


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="RAG Chatbot API", lifespan=lifespan)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query")
def query(req: QueryRequest):
    result = run_query(
        vector_store=_manager.load(),
        query=req.query,
        filter_docs=req.filter_docs,
    )
    return result
