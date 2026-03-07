# src/embeddings.py

import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# Load environment variables from .env file
# This is what reads your OPENAI_API_KEY safely
load_dotenv()

def create_vector_store(chunks: list, persist_directory: str = "vectorstore") -> Chroma:
    # Initialize the embedding model
    # This connects to OpenAI's API using your key from .env
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",  # OpenAI's latest, cheapest embedding model
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    print(f"Creating embeddings for {len(chunks)} chunks...")
    print("This may take a minute — we are calling OpenAI's API for each chunk...")

    # Chroma.from_documents does three things in one call:
    # 1. Takes each chunk's text and sends it to the embedding model
    # 2. Gets back a vector for each chunk
    # 3. Stores both the vector AND the original text in ChromaDB
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )

    print(f"Vector store created with {vector_store._collection.count()} chunks")
    print(f"Saved to '{persist_directory}' folder")

    return vector_store


def load_vector_store(persist_directory: str = "vectorstore") -> Chroma:

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )

    print(f"Loaded existing vector store with {vector_store._collection.count()} chunks")

    return vector_store


if __name__ == "__main__":
    from document_loader import load_documents
    from text_splitter import split_documents

    docs = load_documents("data/sample_docs")

    chunks = split_documents(docs)

    vector_store = create_vector_store(chunks)

    print("\n--- Testing Similarity Search ---")
    query = "What is the population of India?"
    results = vector_store.similarity_search(query, k=3)

    print(f"\nQuery: '{query}'")
    print(f"Found {len(results)} relevant chunks:\n")

    for i, doc in enumerate(results):
        print(f"Result {i+1}:")
        print(f"Content: {doc.page_content[:200]}")
        print(f"Source: {doc.metadata.get('source', 'unknown')}")
        print(f"Page: {doc.metadata.get('page', 'unknown')}")
        print("---")