# 🤖 Enterprise RAG Chatbot

A production-grade **Retrieval-Augmented Generation (RAG)** pipeline that answers questions from custom PDF documents — with mandatory source citations and zero hallucinations on out-of-scope queries.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat-square&logo=langchain&logoColor=white)](https://langchain.com)
[![OpenAI](https://img.shields.io/badge/OpenAI-API-412991?style=flat-square&logo=openai&logoColor=white)](https://openai.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-FF6B35?style=flat-square)](https://trychroma.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)

---

## 🧠 What It Does

Large language models like GPT-3.5 only know what they were trained on — they have no access to private documents, recent data, or domain-specific knowledge bases. They also hallucinate facts confidently, making them unreliable for enterprise use cases.

This RAG chatbot solves both problems:
- **Grounds every answer** in real documents with traceable source citations
- **Refuses to answer** questions outside the provided documents (no hallucination)
- **Cites every fact** with `[Document: filename.pdf | Page: X]`

---

## 🏗️ Architecture

```
PDF Documents
     ↓
PyPDF Loader (DirectoryLoader)
     ↓
RecursiveCharacterTextSplitter
  chunk_size=1000 | chunk_overlap=200
     ↓
OpenAI Embeddings (text-embedding-3-small → 1536-dim vectors)
     ↓
ChromaDB Vector Store (persisted to disk)
     ↓
MMR Retrieval (fetch_k=20 → re-rank → top-5, λ=0.7)
     ↓
format_docs_with_metadata()
  [SOURCE N] | Document: filename | Page: X
     ↓
Strict Citation Prompt → GPT-3.5-turbo
     ↓
Grounded Answer with [Document | Page] Citations
```

---

## ✨ Production Features

| Feature | Details |
|---|---|
| **MMR Retrieval** | Maximum Marginal Relevance balances relevance + diversity; avoids returning near-duplicate chunks |
| **Metadata Citations** | Every chunk is labeled with source filename and page before entering the LLM context |
| **Strict Prompt** | LLM must cite every fact or respond "documents do not contain this information" |
| **Centralized Config** | All settings (chunk size, model names, retrieval params) managed in `config.py` |
| **Error Handling** | Handles missing API key, OpenAI quota exceeded, and missing vector store gracefully |
| **Multi-Document** | Query across multiple PDFs simultaneously; citations identify which document each fact came from |

---

## 📁 Project Structure

```
rag-chatbot/
├── config.py              # Centralized settings (chunk size, models, retrieval params)
├── app.py                 # Streamlit chat UI with error handling
├── ingest.py              # Document ingestion pipeline (run once per data change)
├── data/
│   └── sampledocs/        # Add your PDF files here
├── vectorstore/           # ChromaDB persisted vector store (auto-generated)
├── src/
│   ├── document_loader.py # PDF loading with PyPDF + DirectoryLoader
│   ├── text_splitter.py   # RecursiveCharacterTextSplitter (1000/200)
│   ├── embeddings.py      # OpenAI embeddings + ChromaDB create/load
│   ├── retriever.py       # MMR retriever (k=5, fetch_k=20, λ=0.7)
│   └── llm_chain.py       # format_docs_with_metadata + strict prompt + LCEL chain
├── requirements.txt
└── .env                   # OPENAI_API_KEY (not committed)
```

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/Rajkumar2002-Rk/rag-chatbot.git
cd rag-chatbot
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate      # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up your API key
Create a `.env` file in the project root:
```
OPENAI_API_KEY=your-openai-api-key-here
```

### 5. Add your PDF documents
Place your PDF files in the `data/sampledocs/` folder.

### 6. Build the vector store
```bash
python ingest.py
```

### 7. Run the chatbot
```bash
streamlit run app.py
```

---

## ⚙️ Configuration

All settings are centralized in `config.py`:

```python
CHUNK_SIZE = 1000        # Characters per chunk
CHUNK_OVERLAP = 200      # Overlap between chunks (20% — production standard)
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-3.5-turbo"
LLM_TEMPERATURE = 0      # Deterministic — best for factual Q&A
RETRIEVAL_K = 5          # Chunks returned to LLM
RETRIEVAL_FETCH_K = 20   # Candidates before MMR re-ranking
RETRIEVAL_LAMBDA = 0.7   # Relevance vs. diversity balance
```

---

## 🔬 Key Implementation Details

### Why MMR instead of cosine similarity?
Standard cosine similarity returns the top-k most similar chunks — but they're often near-duplicates from the same paragraph. MMR fetches 20 candidates and re-ranks them to maximize both relevance *and* diversity, ensuring the LLM receives broader context coverage.

### Why 1000/200 chunking?
- `chunk_size=500` cuts sentences mid-thought, losing context
- `chunk_size=1000` preserves full paragraphs and complete ideas
- `chunk_overlap=200` (20%) ensures no information is lost at chunk boundaries — industry standard

### Why format_docs_with_metadata()?
A common mistake in RAG implementations is to strip metadata when formatting documents for the LLM. This function preserves source filename and page number inside the context string, enabling the LLM to cite exactly where each fact came from.

---

## 🛠️ Tech Stack

- **LangChain** — Document loading, text splitting, LCEL chain composition
- **OpenAI API** — `text-embedding-3-small` (embeddings) + `gpt-3.5-turbo` (generation)
- **ChromaDB** — Local vector database with persistent storage
- **Streamlit** — Chat UI with session state and cached resource loading
- **PyPDF** — PDF parsing via `PyPDFLoader`

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
  <sub>Built by <a href="https://rajkumar2002-rk.github.io/Real_Portfolio/">Raj Kumar Nelluri</a> · AI Engineer</sub>
</div>
