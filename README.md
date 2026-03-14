# 🤖 Enterprise RAG Chatbot v2 — Production AI Document Intelligence

A production-grade **Retrieval-Augmented Generation (RAG)** system upgraded with evaluation, observability, multi-document support, hallucination guardrails, and a modular architecture.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-1C3C3C?style=flat-square)](https://langchain.com)
[![OpenAI](https://img.shields.io/badge/OpenAI-API-412991?style=flat-square&logo=openai)](https://openai.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5-FF6B35?style=flat-square)](https://trychroma.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.39-FF4B4B?style=flat-square&logo=streamlit)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker)](https://docker.com)
[![AWS EC2](https://img.shields.io/badge/AWS-EC2-FF9900?style=flat-square&logo=amazonaws)](https://aws.amazon.com/ec2/)

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
     │  [persisted library | in-memory uploads]       │
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
ingestion/pdf_loader.py       ← PyPDFLoader, source metadata patching
   ↓
ingestion/chunking.py         ← RecursiveCharacterTextSplitter (1000/200)
   ↓
ingestion/embedding_pipeline.py ← OpenAI text-embedding-3-small (1536-dim)
   ↓
vectorstore/chroma_manager.py ← ChromaDB (persisted or in-memory)
   ↓
retrieval/retriever.py        ← MMR (fetch_k=20 → rerank → top-5)
   ↓
retrieval/reranker.py         ← Score-based filtering (threshold=0.30)
   ↓  [GUARDRAIL: fallback if low confidence]
api/rag_api.py                ← format_docs_with_metadata() + strict prompt
   ↓
GPT-3.5-turbo                 ← Grounded answer with [SOURCE N: file | Page]
   ↓
monitoring/logger.py          ← JSON log entry to logs/rag_queries.log
```

---

## ✨ Production Improvements (v1 → v2)

| Improvement | What Changed |
|---|---|
| **Modular Architecture** | Business logic moved out of Streamlit into `api/`, `ingestion/`, `retrieval/`, `vectorstore/` modules |
| **Retrieval Evaluation** | `evaluation/run_rag_eval.py` runs benchmark questions and measures hit rate, citation accuracy, latency |
| **Observability** | Structured JSON logging to `logs/rag_queries.log`; real-time metrics dashboard in Streamlit sidebar |
| **Multi-Document Support** | Upload multiple PDFs; select/deselect specific documents for retrieval filtering |
| **Hallucination Guardrails** | Similarity threshold + minimum context length check before calling LLM; fallback response on failure |
| **Score-Based Reranking** | `reranker.py` filters chunks below threshold before LLM context injection |

---

## 📁 Project Structure

```
rag-chatbot-v2/
├── app/
│   └── streamlit_ui.py          # Streamlit UI (library + upload + metrics)
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
├── logs/                        # rag_queries.log written here
├── data/sampledocs/             # Add your PDFs here
├── vectorstore/                 # ChromaDB persisted store
├── ingest.py                    # Top-level ingestion script
├── Dockerfile
├── .dockerignore
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

## 🐳 Setup — Docker

```bash
docker build -t rag-chatbot:latest .

docker volume create rag-vectorstore

docker run -d --name rag-chatbot-app --restart unless-stopped \
  -p 8501:8501 --env-file .env \
  -v rag-vectorstore:/app/vectorstore \
  rag-chatbot:latest

# Index your PDFs into the running container
docker exec rag-chatbot-app python ingest.py
```

Open `http://localhost:8501`

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

**Adding your own benchmark questions** — edit `evaluation/benchmark_questions.json`:
```json
{
  "id": "q011",
  "question": "What is the population of China?",
  "expected_document": "china-wikipedia.pdf",
  "expected_page": 2,
  "reference_answer": "The population of China in 2025 was estimated to be 1,404,890,000.",
  "category": "factual"
}
```

---

## 🔬 Key Implementation Details

### Hallucination Guardrails
Three conditions trigger the fallback response:
1. No documents retrieved from ChromaDB
2. Best similarity score < `SIMILARITY_THRESHOLD` (default: 0.30)
3. Total context length < `MIN_CONTEXT_LENGTH` (default: 50 chars)

### Multi-Document Filtering
ChromaDB's `filter` parameter restricts retrieval to selected documents:
```python
filter={"source": {"$in": ["doc1.pdf", "doc2.pdf"]}}
```

### Structured Logging
Every query writes a JSON line to `logs/rag_queries.log`:
```json
{
  "timestamp": "2026-03-13T14:22:01.123Z",
  "query": "What is the population of India?",
  "response_time_ms": 1240,
  "retrieval_time_ms": 180,
  "num_chunks": 5,
  "source_documents": ["india-wikipedia.pdf"],
  "token_usage_estimate": 312
}
```

---

## ⚙️ Configuration

All settings in `config/settings.py` (overridable via `.env`):

| Variable | Default | Description |
|---|---|---|
| `CHUNK_SIZE` | 1000 | Characters per chunk |
| `CHUNK_OVERLAP` | 200 | Overlap between chunks |
| `RETRIEVAL_K` | 5 | Chunks returned to LLM |
| `RETRIEVAL_FETCH_K` | 20 | MMR candidates before reranking |
| `RETRIEVAL_LAMBDA` | 0.7 | Relevance vs diversity balance |
| `SIMILARITY_THRESHOLD` | 0.30 | Min score to trust a chunk |
| `MIN_CONTEXT_LENGTH` | 50 | Min chars of context before LLM call |

---

## 📄 License

MIT License

<div align="center">
  <sub>Built by <a href="https://rajkumar2002-rk.github.io/Real_Portfolio/">Raj Kumar Nelluri</a> · AI Engineer</sub>
</div>
