"""
ingest.py - Document Ingestion Pipeline

Run this script whenever you:
  - Add new PDFs to the data/ folder
  - Change chunk_size or chunk_overlap in config.py
  - Want to rebuild the vector store from scratch

Usage:
    python ingest.py
"""

import os
import shutil
from dotenv import load_dotenv

from src.document_loader import load_documents
from src.text_splitter import split_documents
from src.embeddings import create_vector_store
from config import DATA_DIR, VECTORSTORE_DIR

load_dotenv()


def ingest():
    print("=" * 50)
    print("RAG Chatbot - Document Ingestion Pipeline")
    print("=" * 50)

    # ── Step 0: Validate data folder ──
    if not os.path.exists(DATA_DIR):
        print(f"\n❌ Error: '{DATA_DIR}/' folder not found.")
        print(f"   Please create it and add your PDF files.")
        return

    pdf_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]
    if not pdf_files:
        print(f"\n❌ Error: No PDF files found in '{DATA_DIR}/'.")
        print(f"   Please add at least one PDF file.")
        return

    print(f"\nFound {len(pdf_files)} PDF file(s):")
    for f in pdf_files:
        print(f"  - {f}")

    # ── Step 1: Clear old vector store ──
    if os.path.exists(VECTORSTORE_DIR):
        print(f"\nRemoving old vector store at '{VECTORSTORE_DIR}/'...")
        shutil.rmtree(VECTORSTORE_DIR)
        print("  Old vector store removed.")

    # ── Step 2: Load documents ──
    print(f"\n[Step 1/3] Loading documents from '{DATA_DIR}/'...")
    documents = load_documents(DATA_DIR)

    # ── Step 3: Split into chunks ──
    print(f"\n[Step 2/3] Splitting documents into chunks...")
    chunks = split_documents(documents)

    # ── Step 4: Embed and store ──
    print(f"\n[Step 3/3] Creating embeddings and saving vector store...")
    create_vector_store(chunks)

    # ── Done ──
    print("\n" + "=" * 50)
    print("✅ Ingestion complete!")
    print(f"   PDFs processed : {len(pdf_files)}")
    print(f"   Pages loaded   : {len(documents)}")
    print(f"   Chunks created : {len(chunks)}")
    print(f"   Vector store   : {VECTORSTORE_DIR}/")
    print("=" * 50)
    print("\nYou can now run the chatbot with:")
    print("   streamlit run app.py")


if __name__ == "__main__":
    ingest()
