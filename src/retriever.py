from langchain_community.vectorstores import Chroma

def get_retriever(vector_store: Chroma, k: int = 3):
    # Convert the vector store into a retriever
    # search_type="similarity" uses cosine similarity
    # k=3 means return top 3 most relevant chunks
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )

    return retriever