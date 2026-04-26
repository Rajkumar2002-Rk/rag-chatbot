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

import chromadb
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import EMBEDDING_MODEL, VECTORSTORE_DIR, OPENAI_API_KEY

# Cosine distance ensures LangChain's relevance scores are computed
# correctly (1 - cosine_distance).  Without this, ChromaDB defaults to
# L2 distance, which produces artificially low scores and triggers
# the hallucination guardrail on perfectly valid content.
_COLLECTION_METADATA = {"hnsw:space": "cosine"}


class ChromaManager:
    """
    Thin wrapper around LangChain's Chroma integration.

    Key design decisions:
      - One collection stores ALL documents; multi-doc filtering is done
        via metadata {"source": filename} at query time.
      - persist=False gives a pure in-memory store — ideal for user uploads
        where nothing should touch disk.
      - All collections use cosine distance so relevance scores are in [0, 1].
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
        self._temp_dir: Optional[str] = None
        self._store: Optional[Chroma] = None

    # ── Internal ──────────────────────────────────────────────────────────────

    def _get_store(self):
        if self._store is None:
            if self.persist:
                client = chromadb.PersistentClient(path=str(self.persist_dir))
            else:
                # Use a temp directory — survives Streamlit reruns better
                # than chromadb.Client() whose in-memory SQLite can go stale.
                import tempfile
                if self._temp_dir is None:
                    self._temp_dir = tempfile.mkdtemp()
                client = chromadb.PersistentClient(path=self._temp_dir)
            self._store = Chroma(
                client=client,
                collection_name=self.collection_name,
                embedding_function=self._embeddings,
                collection_metadata=_COLLECTION_METADATA,
            )
        return self._store

    @property
    def is_healthy(self) -> bool:
        """Check if the underlying store is still accessible."""
        try:
            self._get_store()._collection.count()
            return True
        except Exception:
            return False

    # ── Write ─────────────────────────────────────────────────────────────────

    def add_documents(self, documents: List[Document]) -> None:
        """Add a list of Document chunks to the collection."""
        self._get_store().add_documents(documents)

    def clear(self) -> None:
        """Delete the collection and force re-creation on next access."""
        store = self._get_store()
        store.delete_collection()
        self._store = None   # Force re-init on next access

    def rebuild(self) -> None:
        """
        Delete and re-create the collection with correct settings.

        Use this when migrating from an old collection (e.g. L2 → cosine).
        """
        self.clear()
        self._get_store()    # Re-creates with _COLLECTION_METADATA

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

    def get_all_documents(self) -> List[Document]:
        """
        Return every Document in the collection (for BM25 index building).
        Reconstructs LangChain Document objects from stored content + metadata.
        """
        store = self._get_store()
        results = store.get(include=["documents", "metadatas"])
        docs = []
        for content, meta in zip(
            results.get("documents", []),
            results.get("metadatas", []),
        ):
            docs.append(Document(page_content=content, metadata=meta or {}))
        return docs

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
                kwargs["filter"] = {"source": {"$eq": filter_docs[0]}}
            else:
                kwargs["filter"] = {"source": {"$in": filter_docs}}

        return store.similarity_search_with_relevance_scores(query, **kwargs)
