from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents(documents: list, chunk_size: int = 500, 
                    chunk_overlap: int = 50) -> list:

    # RecursiveCharacterTextSplitter is the most intelligent splitter
    # It tries to split on paragraphs first, then sentences, then words
    # Only splits on characters as a last resort
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,      # use len() to measure chunk size
        add_start_index=True      # adds character position to metadata
    )

    # Split all documents into chunks
    # Metadata from original documents is automatically preserved
    chunks = splitter.split_documents(documents)

    print(f"Split {len(documents)} pages into {len(chunks)} chunks")
    print(f"Settings: chunk_size={chunk_size}, overlap={chunk_overlap}")

    return chunks


if __name__ == "__main__":
    # Import our document loader to test the full pipeline so far
    from document_loader import load_documents

    # Step 1: Load documents
    docs = load_documents("data/sample_docs")

    # Step 2: Split into chunks
    chunks = split_documents(docs)

    # Show first chunk to verify
    if chunks:
        print("\n--- First Chunk Preview ---")
        print("Content:", chunks[0].page_content)
        print("\nMetadata:", chunks[0].metadata)
        print("\nChunk character count:", len(chunks[0].page_content))