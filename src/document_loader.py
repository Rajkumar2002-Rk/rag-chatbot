
import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader

def load_documents(data_path: str) -> list:

    # DirectoryLoader scans the folder and loads all matching files
    # glob="**/*.pdf" means: find all .pdf files in any subfolder
    # loader_cls=PyPDFLoader means: use PyPDFLoader to read each PDF
    loader = DirectoryLoader(
        path=data_path,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )

    # .load() actually reads the files and returns a list of Document objects
    documents = loader.load()

    print(f"Loaded {len(documents)} document pages from '{data_path}'")

    return documents


# This block only runs when you execute this file directly
# It will NOT run when another file imports this module
if __name__ == "__main__":
    docs = load_documents("data/sample_docs")

    # Print first document to verify it loaded correctly
    if docs:
        print("\n--- First Document Preview ---")
        print("Content:", docs[0].page_content[:300])  # first 300 characters
        print("Metadata:", docs[0].metadata)
    else:
        print("No documents found. Add PDF files to data/sample_docs/")