import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from config import LLM_MODEL, LLM_TEMPERATURE

load_dotenv()


# ─────────────────────────────────────────────
# IMPROVEMENT 1: format_docs_with_metadata
# Previously: format_docs() threw away all metadata (source, page)
# Now: injects [SOURCE N], Document name, and Page into the context
# This is what allows the LLM to cite sources accurately
# ─────────────────────────────────────────────
def format_docs_with_metadata(docs):
    """
    Format retrieved chunks with source metadata for citation grounding.

    Instead of just passing raw text, we label each chunk with:
      - SOURCE number (e.g. [SOURCE 1])
      - Document filename
      - Page number

    This lets the LLM say: "According to [Document: india.pdf | Page: 3]..."
    """
    formatted_chunks = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "Unknown")

        # Extract just the filename, not the full path
        filename = source.split("/")[-1] if "/" in source else source

        # ChromaDB pages are 0-indexed, so add 1 for human-readable display
        page_display = page + 1 if isinstance(page, int) else page

        chunk_text = (
            f"[SOURCE {i + 1}]\n"
            f"Document: {filename}\n"
            f"Page: {page_display}\n"
            f"Content: {doc.page_content.strip()}"
        )
        formatted_chunks.append(chunk_text)

    separator = "\n\n" + "=" * 50 + "\n\n"
    return separator.join(formatted_chunks)


# ─────────────────────────────────────────────
# IMPROVEMENT 2: Strict citation-enforcing prompt
# Previously: Basic "Answer using context" prompt — LLM could hallucinate
# Now: STRICT RULES force the LLM to cite every fact or refuse to answer
# ─────────────────────────────────────────────
PROMPT_TEMPLATE = """You are a precise and reliable document assistant.

STRICT RULES:
1. Answer ONLY using the SOURCE DOCUMENTS provided below.
2. NEVER use your training knowledge or make assumptions beyond the documents.
3. For EVERY fact you state, you MUST cite it using this exact format:
   [SOURCE N: filename.pdf | Page: X]
4. If the answer is not found in the provided sources, respond EXACTLY with:
   "The provided documents do not contain information to answer this question."
5. Do NOT combine or guess — cite each fact to its specific source.

SOURCE DOCUMENTS:
{context}

Question: {question}

ANSWER (citations are mandatory for every fact):"""


def build_rag_chain(retriever):
    """
    Build the full RAG chain: retriever → format → prompt → LLM → output.

    Args:
        retriever: LangChain MMR retriever

    Returns:
        LCEL chain ready for .invoke(question)
    """
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in .env file. Please add it.")

        llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            openai_api_key=api_key
        )

        prompt = PromptTemplate(
            template=PROMPT_TEMPLATE,
            input_variables=["context", "question"]
        )

        # LCEL chain using the pipe operator
        chain = (
            {
                "context": retriever | format_docs_with_metadata,
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        return chain

    except Exception as e:
        error_str = str(e).lower()
        if "quota" in error_str or "billing" in error_str or "insufficient" in error_str:
            raise RuntimeError(
                "OpenAI quota exceeded. Please add credits at: platform.openai.com/billing"
            ) from e
        raise


def ask_question(chain, question: str) -> str:
    """
    Ask a question using the RAG chain with error handling.

    Args:
        chain: Built RAG chain
        question: User's question string

    Returns:
        Answer string (or an error message)
    """
    try:
        response = chain.invoke(question)
        return response
    except Exception as e:
        error_str = str(e).lower()
        if "quota" in error_str or "billing" in error_str or "insufficient" in error_str:
            return "⚠️ OpenAI quota exceeded. Please add credits at platform.openai.com/billing"
        return f"⚠️ Error generating response: {str(e)}"
