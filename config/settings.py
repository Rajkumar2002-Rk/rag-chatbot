"""
config/settings.py
──────────────────
Centralized configuration for the RAG system.
All settings loaded from .env with sensible production defaults.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── OpenAI ────────────────────────────────────────────────────────────────────
OPENAI_API_KEY: str   = os.getenv("OPENAI_API_KEY", "")
EMBEDDING_MODEL: str  = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
LLM_MODEL: str        = os.getenv("LLM_MODEL",       "gpt-3.5-turbo")
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0"))

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE: int    = int(os.getenv("CHUNK_SIZE",    "1000"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))

# ── Retrieval ─────────────────────────────────────────────────────────────────
RETRIEVAL_K: int       = int(os.getenv("RETRIEVAL_K",       "5"))
RETRIEVAL_FETCH_K: int = int(os.getenv("RETRIEVAL_FETCH_K", "20"))
RETRIEVAL_LAMBDA: float = float(os.getenv("RETRIEVAL_LAMBDA", "0.7"))

# ── Hallucination Guardrails ──────────────────────────────────────────────────
# Minimum cosine similarity [0–1] for retrieved chunks to be trusted
SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.15"))
# Minimum total characters of context required before calling the LLM
MIN_CONTEXT_LENGTH: int = int(os.getenv("MIN_CONTEXT_LENGTH", "50"))

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR: Path       = Path(__file__).resolve().parent.parent
VECTORSTORE_DIR: str = os.getenv("VECTORSTORE_DIR", str(BASE_DIR / "vectorstore"))
DATA_DIR: str        = os.getenv("DATA_DIR",        str(BASE_DIR / "data" / "sampledocs"))
LOGS_DIR: str        = os.getenv("LOGS_DIR",        str(BASE_DIR / "logs"))
PROMPTS_DIR: str     = str(BASE_DIR / "prompts")

# ── Redis Cache ───────────────────────────────────────────────────────────────
REDIS_URL: str  = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CACHE_TTL: int  = int(os.getenv("CACHE_TTL", "3600"))   # seconds (1 hour)

# ── Ensure runtime directories exist ─────────────────────────────────────────
Path(LOGS_DIR).mkdir(parents=True, exist_ok=True)
Path(VECTORSTORE_DIR).mkdir(parents=True, exist_ok=True)
