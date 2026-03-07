import streamlit as st
from src.embeddings import load_vector_store
from src.retriever import get_retriever
from src.llm_chain import create_rag_chain

# --- Page Configuration ---
# This must be the first Streamlit command in the file
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="centered"
)

# --- Load the RAG chain (runs once when app starts) ---
# @st.cache_resource means this only runs once
# even if the user sends multiple messages
@st.cache_resource
def load_chain():
    vector_store = load_vector_store()
    retriever = get_retriever(vector_store, k=3)
    chain = create_rag_chain(retriever)
    return chain

# --- App Title ---
st.title("🤖 Enterprise RAG Chatbot")
st.caption("Ask questions about your documents")

# --- Initialize chat history ---
# st.session_state persists data between interactions
# without this, chat history disappears on every message
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Display existing chat messages ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat input box at the bottom ---
if prompt := st.chat_input("Ask a question about your documents..."):

    # Add user message to chat history
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get answer from RAG chain
    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):

            # Load chain and get answer
            chain = load_chain()
            answer = chain.invoke(prompt)

        # Display the answer
        st.markdown(answer)

    # Add assistant answer to chat history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })