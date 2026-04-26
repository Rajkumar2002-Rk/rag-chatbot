"""
ingest.py
──────────
Top-level ingestion script.
Run this once to index all PDFs in data/sampledocs/ into ChromaDB.

Usage:
    python ingest.py
    python ingest.py --dir data/sampledocs --collection library
    python ingest.py --rebuild   # delete old collection and re-index (use after config changes)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ingestion.embedding_pipeline import ingest_directory
from vectorstore.chroma_manager import ChromaManager
from config.settings import DATA_DIR


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest PDFs into ChromaDB.")
    parser.add_argument("--dir",        default=DATA_DIR,  help="Directory containing PDFs.")
    parser.add_argument("--collection", default="library", help="ChromaDB collection name.")
    parser.add_argument("--rebuild",    action="store_true",
                        help="Delete existing collection and re-index from scratch.")
    args = parser.parse_args()

    if args.rebuild:
        print(f"Rebuilding collection '{args.collection}' (deleting old data)…")
        manager = ChromaManager(collection_name=args.collection, persist=True)
        manager.rebuild()
        print("Old collection deleted.")

    print(f"Ingesting PDFs from: {args.dir}")
    n = ingest_directory(args.dir, collection_name=args.collection)
    print(f"Done — {n} chunks indexed into collection '{args.collection}'")
