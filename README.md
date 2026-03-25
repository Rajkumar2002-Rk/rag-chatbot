# 🤖 Enterprise RAG Chatbot — Production AI Document Intelligence

A production-grade **Retrieval-Augmented Generation (RAG)** system with evaluation, observability, multi-document support, hallucination guardrails, and a recruiter-friendly UI — containerized with Docker and deployed on AWS EC2.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-1C3C3C?style=flat-square)](https://langchain.com)
[![OpenAI](https://img.shields.io/badge/OpenAI-API-412991?style=flat-square&logo=openai)](https://openai.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5-FF6B35?style=flat-square)](https://trychroma.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.39-FF4B4B?style=flat-square&logo=streamlit)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker)](https://docker.com)
[![AWS EC2](https://img.shields.io/badge/AWS-EC2-FF9900?style=flat-square&logo=amazonaws)](https://aws.amazon.com/ec2/)


## 🚀 Live Demo
👉 **[chatbot.rajkumarai.dev](https://chatbot.rajkumarai.dev)** — deployed on AWS EC2

> Portfolio: [rajkumarai.dev](https://rajkumarai.dev)

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                                │
│           app/streamlit_ui.py  ←→  api/rag_api.py                  │
│     [Library Mode | Upload Mode | Multi-Doc Select | Metrics Panel] │
└────────────────────────────┬────────────────────────────────────────┘
                             │
          ┌──────────────────▼──────────────────┐
          │           api/rag_api.py             │
          │   run_query() — orchestrates all     │
          │   steps below                        │
          └──────────────────┬──────────────────┘
                             │
     ┌───────────────────────▼───────────────────────┐
     │              retrieval/retriever.py            │
     │  retrieve_with_scores() → rerank_by_score()   │
     │  check_retrieval_confidence() [GUARDRAIL]      │
     └───────────────────────┬───────────────────────┘
                             │
     ┌───────────────────────▼───────────────────────┐
     │         vectorstore/chroma_manager.py          │
     │  similarity_search_with_scores()               │
     │  [persisted library | temp-dir uploads]        │
     └───────────────────────┬───────────────────────┘
                             │
     ┌───────────────────────▼───────────────────────┐
     │           LLM (gpt-3.5-turbo)                 │
     │  format_docs_with_metadata() → prompt →       │
     │  [SOURCE N: filename | Page] citations        │
     └───────────────────────┬───────────────────────┘
                             │
     ┌───────────────────────▼───────────────────────┐
     │           monitoring/logger.py                 │
     │  log_query_event() → logs/rag_queries.log     │
     │  monitoring/metrics.py → Streamlit dashboard  │
     └───────────────────────────────────────────────┘
```

---

## 📊 Data Flow

```
PDF Files
   ↓
ingestion/pdf_loader.py         ← PyPDFLoader, source metadata patching
   ↓
ingestion/chunking.py           ← RecursiveCharacterTextSplitter (1000/200)
   ↓
ingestion/embedding_pipeline.py ← OpenAI text-embedding-3-small (1536-dim)
   ↓
vectorstore/chroma_manager.py   ← ChromaDB PersistentClient (persisted or temp-dir)
   ↓
retrieval/retriever.py          ← MMR (fetch_k=20 → rerank → top-5)
   ↓
retrieval/reranker.py           ← Score-based filtering (threshold=0.15)
   ↓  [GUARDRAIL: fallback if low confidence]
api/rag_api.py                  ← format_docs_with_metadata() + strict prompt
   ↓
GPT-3.5-turbo                   ← Grounded answer with [SOURCE N: file | Page]
   ↓
monitoring/logger.py            ← JSON log entry to logs/rag_queries.log
```

---

## ✨ Features

| Feature | Description |
|---|---|
| **Library Mode** | Query pre-indexed PDF documents with multi-doc filtering |
| **Upload Mode** | Upload any PDF and query it instantly (temp-dir storage, no persistence) |
| **Citation-Grounded Answers** | Every response includes `[SOURCE N: filename \| Page]` references |
| **Hallucination Guardrails** | Similarity threshold + min context length check before LLM call |
| **MMR Retrieval** | Maximum Marginal Relevance reduces redundant chunks (fetch_k=20, top-5) |
| **Session Metrics Dashboard** | Real-time latency, token usage, success rate, top documents |
| **Recruiter-Friendly UI** | Welcome guide, clickable example questions, architecture pipeline display |
| **Structured Logging** | Every query logged as JSON to `logs/rag_queries.log` |
| **Modular Architecture** | Business logic split into `api/`, `ingestion/`, `retrieval/`, `vectorstore/` |
| **Docker + AWS EC2** | Containerized with persistent vectorstore volume, live public demo |

---

## 📁 Project Structure

```
rag-chatbot/
├── app/
│   └── streamlit_ui.py          # Streamlit UI (library + upload + metrics + welcome guide)
├── api/
│   └── rag_api.py               # Core RAG logic: run_query(), build_rag_chain()
├── ingestion/
│   ├── pdf_loader.py            # PDF loading + metadata normalization
│   ├── chunking.py              # RecursiveCharacterTextSplitter
│   └── embedding_pipeline.py   # End-to-end ingest pipeline
├── retrieval/
│   ├── retriever.py             # MMR retriever + hallucination guardrails
│   └── reranker.py              # Score-based chunk filtering
├── vectorstore/
│   └── chroma_manager.py        # ChromaDB CRUD + document listing
├── evaluation/
│   ├── run_rag_eval.py          # Evaluation script
│   └── benchmark_questions.json # Ground-truth question set
├── monitoring/
│   ├── logger.py                # Structured JSON logger
│   └── metrics.py               # In-memory metrics tracker
├── prompts/
│   └── rag_prompt.txt           # Strict citation prompt template
├── config/
│   └── settings.py              # Centralized env-based configuration
├── data/sampledocs/             # Library PDFs (Raj_Resume, Attention, GPT-4)
├── vectorstore_data/            # ChromaDB persisted store (volume-mounted)
├── logs/                        # rag_queries.log written here
├── ingest.py                    # Top-level ingestion script
├── Dockerfile
├── .env.example
└── requirements.txt
```

---

## 🚀 Setup — Local

```bash
git clone https://github.com/Rajkumar2002-Rk/rag-chatbot.git
cd rag-chatbot

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt

cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# Add your PDFs to data/sampledocs/

python ingest.py

streamlit run app/streamlit_ui.py
```

---

## 🐳 Setup — Docker (Production)

```bash
# Build the image
docker build -t rag-chatbot-app .

# Run with persistent vectorstore volume
docker run -d \
  --name rag-chatbot-app \
  -p 8501:8501 \
  --env-file .env \
  -v $(pwd)/vectorstore_data:/app/vectorstore_data \
  rag-chatbot-app

# Index PDFs into the running container
docker exec rag-chatbot-app python ingest.py
```

Open `http://localhost:8501`

> **Note:** The `vectorstore_data/` directory is volume-mounted so embeddings persist across container restarts. To re-index with new PDFs, clear the volume with `sudo rm -rf vectorstore_data/*` then re-run `ingest.py`.

---

## 📊 Evaluation

Run the benchmark evaluation against your indexed document library:

```bash
python evaluation/run_rag_eval.py
# or with custom paths:
python evaluation/run_rag_eval.py \
  --questions evaluation/benchmark_questions.json \
  --output evaluation_results.json \
  --collection library
```

**Output metrics:**
- **Retrieval Hit Rate** — % of questions where the expected document was retrieved
- **Citation Accuracy** — % of answers that include a `[SOURCE N: file | Page]` citation
- **Fallback Accuracy** — % of out-of-scope questions correctly handled by guardrail
- **Avg Latency (ms)** — mean end-to-end response time

---

## 🔬 Key Implementation Details

### Hallucination Guardrails
Three conditions trigger the fallback response:
1. No documents retrieved from ChromaDB
2. Best similarity score < `SIMILARITY_THRESHOLD` (default: 0.15)
3. Total context length < `MIN_CONTEXT_LENGTH` (default: 50 chars)

When triggered, the UI shows a helpful tip guiding the user toward specific questions that retrieve well.

### ChromaDB Storage Strategy
- **Library mode** — `PersistentClient(path=VECTORSTORE_DIR)` with Docker volume mount for persistence across restarts
- **Upload mode** — `PersistentClient(path=tempfile.mkdtemp())` for isolated per-session temp storage (avoids ChromaDB 0.5.x EphemeralClient SQLite bug)

### Multi-Document Filtering
ChromaDB's `filter` parameter restricts retrieval to selected documents:
```python
filter={"source": {"$in": ["doc1.pdf", "doc2.pdf"]}}
```

### Structured Logging
Every query writes a JSON line to `logs/rag_queries.log`:
```json
{
  "timestamp": "2026-03-16T05:12:29Z",
  "query": "What programming languages does Raj know?",
  "response_time_ms": 1180,
  "retrieval_time_ms": 192,
  "num_chunks": 1,
  "source_documents": ["Raj_Resume.pdf"],
  "token_usage_estimate": 284
}
```

---

## ⚙️ Configuration

All settings in `config/settings.py` — overridable via `.env`:

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | Required — your OpenAI API key |
| `CHUNK_SIZE` | 1000 | Characters per chunk |
| `CHUNK_OVERLAP` | 200 | Overlap between chunks |
| `RETRIEVAL_K` | 5 | Chunks returned to LLM |
| `RETRIEVAL_FETCH_K` | 20 | MMR candidates before reranking |
| `RETRIEVAL_LAMBDA` | 0.7 | Relevance vs diversity balance |
| `SIMILARITY_THRESHOLD` | 0.15 | Min similarity score to trust a chunk |
| `MIN_CONTEXT_LENGTH` | 50 | Min chars of context before LLM call |
| `VECTORSTORE_DIR` | `vectorstore_data/` | Path to ChromaDB persisted store |

---

## 📄 License

MIT License

<div align="center">
  <sub>Built by <a href="https://rajkumar2002-rk.github.io/Real_Portfolio/">Raj Kumar Nelluri</a> · AI Engineer</sub>
</div>
