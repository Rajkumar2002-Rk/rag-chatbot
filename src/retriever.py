import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import RETRIEVAL_K, RETRIEVAL_FETCH_K, RETRIEVAL_LAMBDA


def get_retriever(vector_store):
    """
    Create an MMR-based retriever from a vector store.

    MMR (Maximum Marginal Relevance) balances:
      - Relevance: how well the chunk matches the query
      - Diversity: avoids returning 5 near-identical chunks

    Args:
        vector_store: Chroma vector store instance

    Returns:
        LangChain retriever
    """
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": RETRIEVAL_K,           # Final chunks returned to the LLM
            "fetch_k": RETRIEVAL_FETCH_K,  # Candidates fetched before MMR ranking
            "lambda_mult": RETRIEVAL_LAMBDA  # 0=max diversity, 1=max relevance
        }
    )
    print(f"Retriever ready: MMR | k={RETRIEVAL_K} | fetch_k={RETRIEVAL_FETCH_K} | lambda={RETRIEVAL_LAMBDA}")
    return retriever
