# RAG Chatbot — Production AI Document Intelligence

A production-grade **Retrieval-Augmented Generation (RAG)** system with query classification, Redis caching, a REST API, hallucination guardrails, citation-grounded answers, and real-time metrics — containerized with Docker and deployed on AWS EC2.

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-1C3C3C?style=flat-square)](https://langchain.com)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--3.5--turbo-412991?style=flat-square&logo=openai)](https://openai.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5-FF6B35?style=flat-square)](https://trychroma.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![Redis](https://img.shields.io/badge/Redis-7-DC382D?style=flat-square&logo=redis)](https://redis.io)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.39-FF4B4B?style=flat-square&logo=streamlit)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker)](https://docker.com)
[![AWS EC2](https://img.shields.io/badge/AWS-EC2-FF9900?style=flat-square&logo=amazonaws)](https://aws.amazon.com/ec2/)

---

## Live Demo

**Chatbot UI:** [chatbot.rajkumarai.dev](https://chatbot.rajkumarai.dev)

**REST API:** [chatbot.rajkumarai.dev/api/query](https://chatbot.rajkumarai.dev/api/query) (POST)

**Swagger Docs:** [chatbot.rajkumarai.dev/api/docs](https://chatbot.rajkumarai.dev/api/docs)

**Portfolio:** [rajkumarai.dev](https://rajkumarai.dev)

Deployed on AWS EC2 (t3.micro) behind nginx with Let's Encrypt SSL.

---

## Features

| Feature | Details |
|---|---|
| **Query Classification** | Rule-based classifier routes each query to an optimized retrieval config (FACTUAL / COMPLEX / AMBIGUOUS / KEYWORD) |
| **Redis Caching** | Normalized query caching with SHA-256 keys, 1-hour TTL — skips caching for fallback/low-confidence responses |
| **REST API** | FastAPI service (`POST /api/query`) with Swagger UI — independently queryable without the Streamlit UI |
| **Citation-Grounded Answers** | Every response includes `[SOURCE N: filename | Page]` references |
| **Hallucination Guardrails** | Similarity threshold + min context length check before the LLM is called |
| **MMR Retrieval** | Maximum Marginal Relevance reduces redundant chunks |
| **Chunk Deduplication** | Deduplicates by `(source, page)` — prevents the same page appearing multiple times in context |
| **Cost Controls** | Rate limiting (20 req/min per IP), input length cap (500 chars), token budget (400 tokens) — all fire before OpenAI is called |
| **Library Mode** | Query pre-indexed PDFs with multi-document filtering |
| **Upload Mode** | Upload any PDF and query it instantly (temp-dir storage, isolated per session) |
| **Real-Time Metrics** | Session latency, token usage, cache hit rate, success rate — tracked in the UI |
| **Structured Logging** | Every query logged as JSON to `logs/rag_queries.log` |

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                            │
│           app/streamlit_ui.py  ←→  api/rag_api.py               │
│    [Library Mode | Upload Mode | Multi-Doc Select | Metrics]     │
└────────────────────────────┬─────────────────────────────────────┘
                             │
          ┌──────────────────▼──────────────────┐
          │          api/server.py (FastAPI)      │
          │  Rate limit (20/min per IP)          │
          │  Input length guard (500 chars)      │
          │  Token budget guard (400 tokens)     │
          │  POST /query → run_query()           │
          │  GET  /health                        │
          └──────────────────┬──────────────────┘
                             │
          ┌──────────────────▼──────────────────┐
          │        cache/redis_cache.py          │
          │  SHA-256 key on normalized query     │
          │  HIT → return cached response        │
          │  MISS → continue pipeline            │
          └──────────────────┬──────────────────┘
                             │
          ┌──────────────────▼──────────────────┐
          │    retrieval/query_classifier.py     │
          │  FACTUAL / COMPLEX / AMBIGUOUS /     │
          │  KEYWORD → sets top_k, fetch_k       │
          └──────────────────┬──────────────────┘
                             │
          ┌──────────────────▼──────────────────┐
          │       retrieval/retriever.py         │
          │  MMR retrieval with per-query config │
          │  check_retrieval_confidence()        │
          └──────────────────┬──────────────────┘
                             │
          ┌──────────────────▼──────────────────┐
          │       retrieval/reranker.py          │
          │  Score filter (threshold=0.15)       │
          │  Dedup by (source, page)             │
          └──────────────────┬──────────────────┘
                             │
          ┌──────────────────▼──────────────────┐
          │     vectorstore/chroma_manager.py    │
          │  ChromaDB — persisted or temp-dir    │
          └──────────────────┬──────────────────┘
                             │
          ┌──────────────────▼──────────────────┐
          │       GPT-3.5-turbo (OpenAI)        │
          │  Strict prompt + citation format     │
          └──────────────────┬──────────────────┘
                             │
          ┌──────────────────▼──────────────────┐
          │      monitoring/logger.py            │
          │  log_query_event() → JSONL log       │
          │  monitoring/metrics.py → UI panel    │
          └─────────────────────────────────────┘
```

---

## Data Flow

```
PDF Files
   ↓
ingestion/pdf_loader.py           ← PyPDFLoader + source metadata patching
   ↓
ingestion/chunking.py             ← RecursiveCharacterTextSplitter (1000/200)
   ↓
ingestion/embedding_pipeline.py   ← OpenAI text-embedding-3-small (1536-dim)
   ↓
vectorstore/chroma_manager.py     ← ChromaDB PersistentClient
   ↓
retrieval/query_classifier.py     ← Classifies query type, sets retrieval config
   ↓
retrieval/retriever.py            ← MMR retrieval with classified config
   ↓
retrieval/reranker.py             ← Score filter + (source, page) deduplication
   ↓  [GUARDRAIL: fallback if low confidence]
api/rag_api.py                    ← format_docs_with_metadata() + strict prompt
   ↓
GPT-3.5-turbo                     ← Grounded answer with [SOURCE N: file | Page]
   ↓
cache/redis_cache.py              ← Cache result for 1 hour
   ↓
monitoring/logger.py              ← JSON log entry to logs/rag_queries.log
```

---

## Project Structure

```
rag-chatbot/
├── app/
│   └── streamlit_ui.py           # Streamlit UI — library, upload, metrics, welcome guide
├── api/
│   ├── rag_api.py                # Core RAG logic: run_query(), build_rag_chain()
│   ├── server.py                 # FastAPI server — rate limiting, input guards, POST /query
│   └── rate_limiter.py           # slowapi Limiter instance (Redis-backed)
├── cache/
│   └── redis_cache.py            # Redis caching with query normalization + stats
├── ingestion/
│   ├── pdf_loader.py             # PDF loading + metadata normalization
│   ├── chunking.py               # RecursiveCharacterTextSplitter
│   └── embedding_pipeline.py    # End-to-end ingest pipeline
├── retrieval/
│   ├── query_classifier.py       # Rule-based classifier → RetrievalConfig
│   ├── retriever.py              # MMR retriever + hallucination guardrails
│   └── reranker.py               # Score filtering + (source, page) deduplication
├── vectorstore/
│   └── chroma_manager.py         # ChromaDB CRUD + document listing
├── monitoring/
│   ├── logger.py                 # Structured JSON logger
│   └── metrics.py                # MetricsTracker (latency, tokens, cache hit rate)
├── evaluation/
│   └── run_rag_eval.py           # Evaluation script
├── prompts/
│   └── rag_prompt.txt            # Strict citation prompt template
├── config/
│   └── settings.py               # All config from .env
├── data/sampledocs/              # Library PDFs
├── vectorstore_data/             # ChromaDB persisted store (volume-mounted)
├── logs/                         # rag_queries.log written here
├── ingest.py                     # Top-level ingestion script
├── docker-compose.yml            # 3 services: redis, app (Streamlit), api (FastAPI)
├── Dockerfile
├── .env.example
└── requirements.txt
```

---

## Setup — Local

```bash
git clone https://github.com/Rajkumar2002-Rk/rag-chatbot.git
cd rag-chatbot

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt

cp .env.example .env
# Add your OPENAI_API_KEY to .env

# Add PDFs to data/sampledocs/
python ingest.py

# Run Streamlit UI
streamlit run app/streamlit_ui.py

# Run FastAPI (separate terminal)
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

---

## Setup — Docker Compose (Production)

```bash
cp .env.example .env
# Add your OPENAI_API_KEY to .env

docker-compose up -d
```

This starts three services:
- **redis** — Redis 7 (caching)
- **app** — Streamlit UI on port 8501
- **api** — FastAPI on port 8000

Open `http://localhost:8501` for the UI or hit `http://localhost:8000/query` for the API.

> The `vectorstore_data/` and `logs/` directories are volume-mounted so data persists across restarts.

---

## REST API

### `POST /api/query`

```bash
curl -X POST https://chatbot.rajkumarai.dev/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the attention mechanisms described in the paper?"}'
```

**Request body:**
```json
{
  "query": "string",
  "filter_docs": ["doc1.pdf", "doc2.pdf"]  // optional
}
```

**Response:**
```json
{
  "answer": "The paper describes...",
  "sources": ["Attention Is All You Need.pdf"],
  "query_type": "COMPLEX",
  "cache_hit": false,
  "response_time_ms": 1240,
  "num_chunks": 7
}
```

Swagger UI: [chatbot.rajkumarai.dev/api/docs](https://chatbot.rajkumarai.dev/api/docs)

---

## Query Classification

Each query is classified before retrieval, with a retrieval config tuned per type:

| Type | top_k | fetch_k | Notes |
|---|---|---|---|
| `FACTUAL` | 3 | 10 | Short, specific lookups |
| `COMPLEX` | 7 | 25 | Multi-part questions |
| `AMBIGUOUS` | 5 | 15 | Query expansion applied |
| `KEYWORD` | 5 | 15 | Keyword/entity lookups |

---

## Redis Caching

- **Key:** SHA-256 hash of normalized query + `filter_docs`
- **Normalization:** lowercase, collapse whitespace, strip punctuation, strip filler prefixes ("what is", "tell me about", etc.)
- **TTL:** 1 hour (configurable via `CACHE_TTL`)
- **Skips caching:** fallback responses, low-confidence results, errors
- **Stats tracked:** hits, misses, sets, skipped — surfaced in the UI metrics panel

---

## Hallucination Guardrails

Three conditions trigger a fallback response instead of an LLM call:

1. No documents retrieved from ChromaDB
2. Best similarity score < `SIMILARITY_THRESHOLD` (default: `0.15`)
3. Total context length < `MIN_CONTEXT_LENGTH` (default: `50` chars)

When triggered, the user sees a helpful message guiding them toward better queries.

---

## Configuration

All settings in `config/settings.py`, overridable via `.env`:

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | Required |
| `REDIS_URL` | `redis://redis:6379/0` | Redis connection URL |
| `CACHE_TTL` | `3600` | Cache TTL in seconds |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `RETRIEVAL_K` | `5` | Chunks returned to LLM |
| `RETRIEVAL_FETCH_K` | `20` | MMR candidates before reranking |
| `RETRIEVAL_LAMBDA` | `0.7` | Relevance vs diversity balance |
| `SIMILARITY_THRESHOLD` | `0.15` | Min score to trust a chunk |
| `MIN_CONTEXT_LENGTH` | `50` | Min chars of context before LLM call |
| `RATE_LIMIT` | `20/minute` | Max requests per IP per window |
| `MAX_QUERY_LENGTH` | `500` | Max query length in characters |
| `MAX_INPUT_TOKENS` | `400` | Max query tokens (tiktoken) before LLM |
| `VECTORSTORE_DIR` | `vectorstore_data/` | ChromaDB persist path |

---

## Cost Controls

Three guards fire on every `/query` request **before** OpenAI is ever called:

| Guard | Limit | Response on breach |
|---|---|---|
| **Rate limiting** | 20 requests/minute per IP (slowapi + Redis) | HTTP 429 |
| **Input length** | 500 characters max | HTTP 400 |
| **Token budget** | 400 tokens max (tiktoken `cl100k_base`) | HTTP 400 |

All limits are configurable via `.env` — see `RATE_LIMIT`, `MAX_QUERY_LENGTH`, `MAX_INPUT_TOKENS`.

---

## Structured Logging

Every query writes a JSON line to `logs/rag_queries.log`:

```json
{
  "timestamp": "2026-03-28T10:22:11Z",
  "query": "What programming languages does Raj know?",
  "query_type": "FACTUAL",
  "cache_hit": false,
  "response_time_ms": 1180,
  "retrieval_time_ms": 192,
  "num_chunks": 3,
  "source_documents": ["Raj_Resume.pdf"],
  "token_usage_estimate": 284
}
```

---

## Docker Hub

Pre-built image: [hub.docker.com/r/raja1566/rag-chatbot](https://hub.docker.com/r/raja1566/rag-chatbot)

```bash
docker pull raja1566/rag-chatbot:latest
```

---

## License

MIT License

<div align="center">
  <sub>Built by <a href="https://rajkumarai.dev">Raj Kumar Nelluri</a> · AI Engineer</sub>
</div>
