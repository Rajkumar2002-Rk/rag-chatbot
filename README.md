# 🤖 Enterprise RAG Chatbot

A production-ready Retrieval-Augmented Generation (RAG) chatbot that answers questions from your own documents with source citations. Built with LangChain, OpenAI, ChromaDB, and Streamlit.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.2+-green)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--3.5--turbo-orange)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector--Store-purple)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red)

---

## 📌 What This Project Does

Most AI chatbots only know what they were trained on. This chatbot reads **your own documents** and answers questions based on them — accurately, with source citations, and without hallucinating.

**Example:**
- Upload a company policy PDF → Ask "What is the refund policy?" → Get an exact answer with page number
- Upload a research paper → Ask "What were the key findings?" → Get a grounded, cited answer

---

## 🏗️ Architecture

```
PDF Documents
      ↓
Document Loader (PyPDF)
      ↓
Text Splitter (500 char chunks, 50 char overlap)
      ↓
Embedding Model (OpenAI text-embedding-3-small)
      ↓
Vector Store (ChromaDB)
      ↓
User Question → Query Vector → Similarity Search (Top 3 chunks)
      ↓
Prompt Template + Retrieved Chunks → GPT-3.5-turbo
      ↓
Grounded Answer with Source Citation
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.10+ | Core language |
| LangChain | RAG pipeline orchestration |
| OpenAI API | Embeddings + LLM (GPT-3.5-turbo) |
| ChromaDB | Local vector database |
| PyPDF | PDF text extraction |
| Streamlit | Web interface |
| python-dotenv | Secure API key management |

---

## 📁 Project Structure

```
rag-chatbot/
│
├── data/
│   └── sample_docs/          # Add your PDF documents here
│
├── src/
│   ├── document_loader.py    # Loads and parses PDF files
│   ├── text_splitter.py      # Splits documents into chunks
│   ├── embeddings.py         # Creates and stores embeddings
│   ├── retriever.py          # Retrieves relevant chunks
│   └── llm_chain.py          # Connects retriever to LLM
│
├── vectorstore/              # ChromaDB data (auto-generated)
├── app.py                    # Streamlit web interface
├── ingest.py                 # Document ingestion pipeline
├── requirements.txt          # Project dependencies
└── .env                      # API keys (never commit this)
```

---

## ⚙️ Installation

**1. Clone the repository**
```bash
git clone https://github.com/Rajkumar2002-Rk/rag-chatbot.git
cd rag-chatbot
```

**2. Create and activate virtual environment**
```bash
python -m venv venv
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Set up your API key**

Create a `.env` file in the root folder:
```
OPENAI_API_KEY=your_openai_api_key_here
```
Get your API key at: https://platform.openai.com

---

## 🚀 Usage

**Step 1 — Add your documents**

Place any PDF files inside `data/sample_docs/`

**Step 2 — Run ingestion**
```bash
python ingest.py
```
This processes your documents, creates embeddings, and builds the vector store. Run this once, or whenever you add new documents.

**Step 3 — Start the chatbot**
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` and start asking questions.

---

## 💡 Key Features

- **Multi-document support** — Load and query multiple PDFs simultaneously
- **Semantic search** — Finds relevant content even when exact words don't match
- **Source citations** — Every answer includes the source document and page number
- **Hallucination prevention** — Prompt engineering ensures answers stay grounded in documents
- **Persistent vector store** — Embeddings saved to disk, no re-processing on restart
- **Clean web UI** — Chat interface with message history via Streamlit

---

## 🧠 How It Works

**Ingestion Phase (runs once):**
1. PDFs are loaded and text is extracted page by page
2. Text is split into 500-character chunks with 50-character overlap
3. Each chunk is converted to a 1536-dimensional vector using OpenAI embeddings
4. Vectors and original text are stored in ChromaDB

**Query Phase (every question):**
1. User's question is converted to a query vector using the same embedding model
2. ChromaDB finds the top 3 most similar chunks using cosine similarity
3. Retrieved chunks are injected into a prompt template
4. GPT-3.5-turbo generates a grounded answer with source citations

---

## 🔍 Example Queries

```
✅ "What is the population of India?"
✅ "When was Python founded?"
✅ "What religions are practiced in India?"
✅ "What are the benefits of using Python?"
❌ "What is the weather today?" → "I don't have enough information" (correct behavior)
```

---

## 🚧 Future Improvements

- [ ] Add support for Word documents (.docx) and web URLs
- [ ] Implement RAGAS evaluation framework for answer quality metrics
- [ ] Replace ChromaDB with Pinecone for cloud-scale deployment
- [ ] Add re-ranking layer for improved retrieval precision
- [ ] Implement conversation memory for follow-up questions
- [ ] Add document upload UI directly in the Streamlit interface
- [ ] Deploy to AWS/GCP with authentication for multi-user access

---

## 📊 Skills Demonstrated

- Retrieval-Augmented Generation (RAG) pipeline design
- Vector embeddings and semantic search
- LangChain framework and LCEL chains
- Prompt engineering for hallucination prevention
- Vector database management (ChromaDB)
- OpenAI API integration
- Python modular code architecture
- Streamlit web application development

---

## 👤 Author

**Raj Kumar**
- GitHub: [@Rajkumar2002-Rk](https://github.com/Rajkumar2002-Rk)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).