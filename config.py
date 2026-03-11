# config.py - Centralized configuration for RAG Chatbot
# Edit these values to tune your chatbot's behavior

# --- Chunking ---
CHUNK_SIZE = 1000        # Characters per chunk (was 500)
CHUNK_OVERLAP = 200      # Overlap between chunks (was 50)

# --- OpenAI Models ---
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-3.5-turbo"
LLM_TEMPERATURE = 0      # 0 = deterministic, no creativity (best for factual Q&A)

# --- Retrieval ---
RETRIEVAL_K = 5          # Number of final chunks to return
RETRIEVAL_FETCH_K = 20   # Candidates fetched before MMR re-ranking
RETRIEVAL_LAMBDA = 0.7   # 0=max diversity, 1=max relevance (0.7 = balanced)

# --- Paths ---
VECTORSTORE_DIR = "vectorstore"
DATA_DIR = "data/sample_docs"
