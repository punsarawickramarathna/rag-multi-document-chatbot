import streamlit as st
import os
from utils.rag_pipeline import *

st.set_page_config(
    page_title="Multi Document RAG Chatbot",
    page_icon="📚",
    layout="wide"
)

# ---------- TITLE ----------
st.markdown(
"""
# 📚 Multi-Document RAG Chatbot
Ask questions from multiple PDF documents using AI
"""
)

st.divider()

# ---------- SESSION STATE ----------
if "chain" not in st.session_state:
    st.session_state.chain = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------- SIDEBAR ----------
with st.sidebar:

    st.header("📂 Upload Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type="pdf",
        accept_multiple_files=True
    )

    if uploaded_files:

        os.makedirs("data", exist_ok=True)

        for file in uploaded_files:
            with open(os.path.join("data", file.name), "wb") as f:
                f.write(file.getbuffer())

        st.success(" Files uploaded")

        with st.spinner("Processing documents..."):

            documents = load_documents("data")
            chunks = split_documents(documents)
            vectordb = create_vector_db(chunks)

            st.session_state.chain = create_chain(vectordb)

        st.success(" RAG System Ready!")

    st.divider()

    st.markdown("### ℹ️ Instructions")
    st.write("1. Upload one or more PDF files")
    st.write("2. Ask questions about the documents")
    st.write("3. AI will answer using document context")

# ---------- CHAT AREA ----------
st.subheader("💬 Chat with your documents")

question = st.chat_input("Ask a question about your PDFs...")

if question and st.session_state.chain:

    with st.spinner("Thinking..."):

        result = st.session_state.chain({"question": question})

        answer = result["answer"].split("Answer:")[-1].strip()
        sources = result["source_documents"]

    # Save history
    st.session_state.chat_history.append(("user", question))
    st.session_state.chat_history.append(("assistant", answer))

# ---------- DISPLAY CHAT ----------
for role, message in st.session_state.chat_history:

    if role == "user":
        with st.chat_message("user"):
            st.write(message)

    else:
        with st.chat_message("assistant"):
            st.write(message)

# ---------- SOURCE DOCUMENTS ----------
if question and st.session_state.chain:

    with st.expander("📑 View Sources"):

        for doc in sources:
            st.write("📄", doc.metadata["source"])