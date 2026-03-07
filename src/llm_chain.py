import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

def create_rag_chain(retriever):

    # Initialize the language model
    # temperature=0 means deterministic answers (no randomness)
    # good for factual Q&A where consistency matters
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    # This is our prompt template
    # {context} and {question} are placeholders filled at query time
    prompt_template = """
You are a helpful assistant. Answer the question based ONLY on the 
context provided below. If the answer is not found in the context, 
say "I don't have enough information to answer this question."

Always be concise and accurate. At the end of your answer, 
mention which page the information came from if available.
If page number is not available, just mention the document name.

Context:
{context}

Question: {question}

Answer:"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # Helper function to format retrieved chunks into a single string
    # Each chunk's text is joined with a separator
    def format_docs(docs):
        return "\n\n---\n\n".join([doc.page_content for doc in docs])

    # Build the chain using LangChain's pipe operator |
    # This is called LCEL (LangChain Expression Language)
    # Read it left to right: retrieve → format → prompt → llm → parse output
    rag_chain = (
        {
            "context": retriever | format_docs,  # retrieve chunks, format them
            "question": RunnablePassthrough()     # pass question through unchanged
        }
        | prompt          # inject context + question into prompt template
        | llm             # send filled prompt to GPT
        | StrOutputParser()  # extract the text string from GPT's response
    )

    return rag_chain


if __name__ == "__main__":
    from embeddings import load_vector_store
    from retriever import get_retriever

    # Load existing vector store (no API call, reads from disk)
    vector_store = load_vector_store()

    # Create retriever
    retriever = get_retriever(vector_store, k=3)

    # Create RAG chain
    rag_chain = create_rag_chain(retriever)

    # Test with a question
    print("RAG Chain ready. Testing...\n")

    question = "Tell me abouth the India and its culture."
    print(f"Question: {question}")
    print("\nAnswer:")

    # .invoke() runs the entire chain end to end
    answer = rag_chain.invoke(question)
    print(answer)