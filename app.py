import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Page config (must be first Streamlit call) ──
st.set_page_config(
    page_title="Enterprise RAG Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Enterprise RAG Chatbot")
st.caption("Ask questions about your uploaded documents")


# ── Load the RAG chain (cached so it only loads once) ──
@st.cache_resource
def load_chain():
    """Load vector store, retriever, and LLM chain. Cache to avoid reloading on every message."""
    try:
        from src.embeddings import load_vector_store
        from src.retriever import get_retriever
        from src.llm_chain import build_rag_chain

        vector_store = load_vector_store()
        retriever = get_retriever(vector_store)
        chain = build_rag_chain(retriever)
        return chain

    except FileNotFoundError as e:
        st.error(
            f"⚠️ **Vector store not found.**\n\n"
            f"Please run this command in your terminal first:\n"
            f"```\npython ingest.py\n```\n\n"
            f"Details: {e}"
        )
        st.stop()

    except ValueError as e:
        st.error(f"⚠️ **Configuration error:** {e}")
        st.stop()

    except RuntimeError as e:
        st.error(f"⚠️ {e}")
        st.stop()

    except Exception as e:
        st.error(f"⚠️ **Unexpected error loading chatbot:** {e}")
        st.stop()


# ── Initialize ──
chain = load_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Display chat history ──
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ── Handle new user input ──
if prompt := st.chat_input("Ask a question about your documents..."):

    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and show assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            try:
                response = chain.invoke(prompt)
            except Exception as e:
                error_str = str(e).lower()
                if "quota" in error_str or "billing" in error_str or "insufficient" in error_str:
                    response = "⚠️ OpenAI quota exceeded. Please add credits at platform.openai.com/billing"
                else:
                    response = f"⚠️ Error: {str(e)}"

        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
