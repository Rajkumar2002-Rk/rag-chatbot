# Run this script whenever you add new documents to data/sample_docs/
# This rebuilds the vector store with all current documents
import os
import shutil
from src.document_loader import load_documents
from src.text_splitter import split_documents
from src.embeddings import create_vector_store

def ingest():
    """
    Full ingestion pipeline:
    1. Load all PDFs from data/sample_docs/
    2. Split into chunks
    3. Create embeddings and store in ChromaDB
    """

    DOCS_PATH = "data/sample_docs"
    VECTOR_STORE_PATH = "vectorstore"

    # Safety check — make sure documents folder exists
    if not os.path.exists(DOCS_PATH):
        print(f"Error: '{DOCS_PATH}' folder not found.")
        print("Create the folder and add PDF files before running ingestion.")
        return

    # If vector store already exists, delete it first
    # This ensures we start fresh with all current documents
    if os.path.exists(VECTOR_STORE_PATH):
        print(f"Removing existing vector store...")
        shutil.rmtree(VECTOR_STORE_PATH)
        print("Done.")

    # Step 1: Load documents
    print("\nStep 1: Loading documents...")
    documents = load_documents(DOCS_PATH)

    if not documents:
        print("No documents found. Add PDF files to data/sample_docs/")
        return

    # Step 2: Split into chunks
    print("\nStep 2: Splitting into chunks...")
    chunks = split_documents(documents)

    # Step 3: Create vector store
    print("\nStep 3: Creating embeddings and storing in ChromaDB...")
    create_vector_store(chunks, persist_directory=VECTOR_STORE_PATH)

    print("\n✅ Ingestion complete!")
    print(f"Your chatbot is ready. Run: streamlit run app.py")


if __name__ == "__main__":
    ingest()