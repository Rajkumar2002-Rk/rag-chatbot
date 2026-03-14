"""
vectorstore/chroma_manager.py
──────────────────────────────
ChromaDB management layer.

Supports:
  - Persisted collections (library mode)
  - In-memory collections (upload mode — nothing written to disk)
  - Multi-document filtering via metadata
  - Listing all documents in a collection
"""
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import EMBEDDING_MODEL, VECTORSTORE_DIR, OPENAI_API_KEY
import chromadb
from langchain_chroma import Chroma

class ChromaManager:
    """
    Thin wrapper around LangChain's Chroma integration.

    Key design decisions:
      - One collection stores ALL documents; multi-doc filtering is done
        via metadata {"source": filename} at query time.
      - persist=False gives a pure in-memory store — ideal for user uploads
        where nothing should touch disk.
    """

    def __init__(
        self,
        collection_name: str = "rag_documents",
        persist: bool         = True,
        persist_dir: Optional[str] = None,
    ) -> None:
        self.collection_name = collection_name
        self.persist         = persist
        self.persist_dir     = persist_dir or VECTORSTORE_DIR

        self._embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            openai_api_key=OPENAI_API_KEY,
        )
        self._store: Optional[Chroma] = None

    # ── Internal ──────────────────────────────────────────────────────────────

    def _get_store(self):
        if self._store is None:
          if self.persist_dir is None:
            # Upload/in-memory mode
            client = chromadb.EphemeralClient()
          else:
            # Library/persistent mode
            client = chromadb.PersistentClient(path=str(self.persist_dir))
          self._store = Chroma(
            client=client,
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
         )
        return self._store

    # ── Write ─────────────────────────────────────────────────────────────────

    def add_documents(self, documents: List[Document]) -> None:
        """Add a list of Document chunks to the collection."""
        self._get_store().add_documents(documents)

    def clear(self) -> None:
        """Delete all documents from the collection."""
        store = self._get_store()
        store.delete_collection()
        self._store = None   # Force re-init on next access

    # ── Read ──────────────────────────────────────────────────────────────────

    def load(self) -> Chroma:
        """Return the underlying Chroma store (for use with LangChain retrievers)."""
        return self._get_store()

    def get_document_list(self) -> List[str]:
        """
        Return a sorted list of unique document filenames in this collection.
        Used to populate the multi-document selector in the UI.
        """
        store   = self._get_store()
        results = store.get(include=["metadatas"])
        sources = {
            meta["source"]
            for meta in results.get("metadatas", [])
            if meta and "source" in meta
        }
        return sorted(sources)

    def get_chunk_count(self) -> int:
        """Return total number of chunks stored in the collection."""
        return self._get_store()._collection.count()

    # ── Search ────────────────────────────────────────────────────────────────

    def similarity_search_with_scores(
        self,
        query: str,
        k: int = 5,
        filter_docs: Optional[List[str]] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Run similarity search and return (Document, score) pairs.
        Score is cosine relevance in [0, 1] — higher is more relevant.

        Args:
            filter_docs: If provided, restrict search to these filenames.
        """
        store  = self._get_store()
        kwargs: dict = {"k": k}

        if filter_docs:
            if len(filter_docs) == 1:
                kwargs["filter"] = {"source": filter_docs[0]}
            else:
                kwargs["filter"] = {"source": {"$in": filter_docs}}

        return store.similarity_search_with_relevance_scores(query, **kwargs)
