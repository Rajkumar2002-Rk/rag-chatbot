"""
ingest.py
──────────
Top-level ingestion script.
Run this once to index all PDFs in data/sampledocs/ into ChromaDB.

Usage:
    python ingest.py
    python ingest.py --dir data/sampledocs --collection library
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ingestion.embedding_pipeline import ingest_directory
from config.settings import DATA_DIR


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest PDFs into ChromaDB.")
    parser.add_argument("--dir",        default=DATA_DIR,  help="Directory containing PDFs.")
    parser.add_argument("--collection", default="library", help="ChromaDB collection name.")
    args = parser.parse_args()

    print(f"Ingesting PDFs from: {args.dir}")
    n = ingest_directory(args.dir, collection_name=args.collection)
    print(f"✅ Done — {n} chunks indexed into collection '{args.collection}'")
