import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

load_dotenv()


st.set_page_config(
    page_title="Enterprise RAG Chatbot",
    page_icon="🤖",
    layout="centered"
)


# LIBRARY MODE — load the pre-built vectorstore from disk
#
# @st.cache_resource means this function runs ONCE and the result
# is reused for every user. We don't want to reload ChromaDB from
# disk on every message — that would be slow and expensive.

@st.cache_resource
def load_library_chain():
    """Load the pre-built vectorstore and return a ready-to-use RAG chain."""
    try:
        from src.embeddings import load_vector_store
        from src.retriever import get_retriever
        from src.llm_chain import build_rag_chain

        vector_store = load_vector_store()
        retriever = get_retriever(vector_store)
        chain = build_rag_chain(retriever)
        return chain, None   # (chain, error)

    except FileNotFoundError as e:
        return None, str(e)
    except ValueError as e:
        return None, str(e)
    except RuntimeError as e:
        return None, str(e)
    except Exception as e:
        return None, str(e)



def process_uploaded_pdf(uploaded_file):
    """
    Process a user-uploaded PDF and return a RAG chain.
    The vectorstore is created in memory — nothing persists to disk.
    """
    try:
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_chroma import Chroma
        from langchain_openai import OpenAIEmbeddings
        from src.text_splitter import split_documents
        from src.retriever import get_retriever
        from src.llm_chain import build_rag_chain
        from config import EMBEDDING_MODEL

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in .env file.")

        # Step 1: Save uploaded bytes to a temporary file on disk
        # PyPDFLoader requires a file path — it can't read from memory directly
        # delete=False so we can pass the path to PyPDFLoader before deleting it
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        # Step 2: Load the PDF
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()

        # Step 3: Delete the temp file immediately — we have the content in memory now
        os.unlink(tmp_path)

        # Fix: replace temp path with real uploaded filename in metadata
        for doc in documents:
            doc.metadata["source"] = uploaded_file.name

        if not documents:
            raise ValueError("Could not extract any text from the PDF. Is it a scanned image?")

        # Step 4: Split into chunks
        chunks = split_documents(documents)

        # Step 5: Create embeddings and in-memory vectorstore
        # NOTE: No persist_directory → ChromaDB stays in RAM only
        embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            openai_api_key=api_key
        )
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings
            # No persist_directory = in-memory only
        )

        # Step 6: Build and return the RAG chain
        retriever = get_retriever(vector_store)
        chain = build_rag_chain(retriever)

        return chain, len(chunks), None   # (chain, chunk_count, error)

    except Exception as e:
        error_str = str(e).lower()
        if "quota" in error_str or "billing" in error_str or "insufficient" in error_str:
            return None, 0, "OpenAI quota exceeded. Please add credits at platform.openai.com/billing"
        return None, 0, str(e)


with st.sidebar:
    st.title("⚙️ Settings")
    st.divider()

    mode = st.radio(
        "Choose Mode",
        ["📚 Library Documents", "📤 Upload Your PDF"],
        help="Library: search pre-loaded documents. Upload: bring your own PDF."
    )

    st.divider()

    # Upload mode — show file uploader in sidebar
    if mode == "📤 Upload Your PDF":
        uploaded_file = st.file_uploader(
            "Upload a PDF file",
            type=["pdf"],
            help="Your PDF is processed in memory and not stored on the server."
        )
        if uploaded_file:
            st.success(f"📄 {uploaded_file.name}")
            st.caption("🔒 Processed in memory · Not stored on server")
    else:
        uploaded_file = None

    st.divider()
    st.caption("Built with LangChain · OpenAI · ChromaDB")
    st.caption("[View on GitHub](https://github.com/Rajkumar2002-Rk/rag-chatbot)")


st.title("🤖 Enterprise RAG Chatbot")

if mode == "📚 Library Documents":

    st.caption("Ask questions about the pre-loaded document library.")

    # Load library chain (cached — only runs once for all users)
    chain, error = load_library_chain()

    if error:
        if "not found" in error.lower() or "ingest" in error.lower():
            st.error(
                "⚠️ **Vector store not found.**\n\n"
                "Please run this in your terminal:\n```\npython ingest.py\n```"
            )
        else:
            st.error(f"⚠️ {error}")
        st.stop()

    # ── Reset chat history when switching to this mode ──
    if st.session_state.get("current_mode") != "library":
        st.session_state.messages = []
        st.session_state.current_mode = "library"

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle new question
    if prompt := st.chat_input("Ask a question about the library documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents..."):
                try:
                    response = chain.invoke(prompt)
                except Exception as e:
                    error_str = str(e).lower()
                    if "quota" in error_str or "billing" in error_str:
                        response = "⚠️ OpenAI quota exceeded. Please add credits at platform.openai.com/billing"
                    else:
                        response = f"⚠️ Error: {str(e)}"
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})


# ── Mode: Upload Your PDF ────────────────────────────────────
elif mode == "📤 Upload Your PDF":

    st.caption("Upload any PDF and ask questions about it.")

    if not uploaded_file:
        # No file uploaded yet — show instructions
        st.info(
            "👈 Upload a PDF from the sidebar to get started.\n\n"
            "Your document is processed in memory and **never stored on the server**."
        )
        st.stop()

    # ── Detect if a NEW file was uploaded ──
    # We track the filename in session_state.
    # If the filename changed, we rebuild the chain from scratch.
    if st.session_state.get("uploaded_filename") != uploaded_file.name:

        with st.spinner(f"Processing **{uploaded_file.name}**... this takes ~15 seconds"):
            chain, chunk_count, error = process_uploaded_pdf(uploaded_file)

        if error:
            st.error(f"⚠️ Failed to process PDF: {error}")
            st.stop()

        # Store chain and metadata in session_state
        st.session_state.upload_chain = chain
        st.session_state.uploaded_filename = uploaded_file.name
        st.session_state.upload_chunk_count = chunk_count
        st.session_state.messages = []          # Reset chat for new document
        st.session_state.current_mode = "upload"
        st.rerun()

    # File is already processed — retrieve chain from session_state
    chain = st.session_state.get("upload_chain")

    if not chain:
        st.error("⚠️ Something went wrong processing the PDF. Please try uploading again.")
        st.stop()

    # Show success banner
    st.success(
        f"✅ **{st.session_state.uploaded_filename}** ready — "
        f"{st.session_state.get('upload_chunk_count', '?')} chunks indexed"
    )

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle new question
    if prompt := st.chat_input(f"Ask a question about {st.session_state.uploaded_filename}..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching your document..."):
                try:
                    response = chain.invoke(prompt)
                except Exception as e:
                    error_str = str(e).lower()
                    if "quota" in error_str or "billing" in error_str:
                        response = "⚠️ OpenAI quota exceeded. Please add credits at platform.openai.com/billing"
                    else:
                        response = f"⚠️ Error: {str(e)}"
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})