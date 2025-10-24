# -*- coding: utf-8 -*-
"""
BasisQuest Streamlit App
------------------------
A clean, professional web interface for the BasisQuest RAG chatbot.
Displays document loading progress and embedding status dynamically.
"""

import streamlit as st
from BasisQuest import (
    load_documents, build_index, get_embeddings_batch,
    rerank, chat, update_history_queue, DOCS_FOLDER,
    TOP_K, RERANK_TOP_M
)
from collections import deque
import numpy as np
import time

# -----------------------------
# Streamlit page setup
# -----------------------------
st.set_page_config(page_title="BasisQuest", layout="centered")

st.title("BasisQuest - AI Knowledge Assistant for SAP")
st.write("Ask questions about SAP Basis and enterprise automation documentation.")

# -----------------------------
# Custom document loader with progress
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_and_index_with_progress():
    progress_text = st.empty()
    bar = st.progress(0)

    # Step 1: Load documents
    progress_text.text("Loading documents...")
    docs = load_documents(DOCS_FOLDER)

    # Simulate progress for better UX
    for i in range(10):
        time.sleep(0.05)
        bar.progress(int((i + 1) * 5))

    # Step 2: Build embeddings and index
    progress_text.text("Building embeddings and FAISS index...")

    with st.spinner("Embedding documents... this may take a while."):
        index, all_embs = build_index(docs)

    bar.progress(100)
    progress_text.text(f"Documents loaded, total chunks: {len(docs)}")
    return docs, index, all_embs

# -----------------------------
# Load documents once
# -----------------------------
with st.spinner("Initializing and indexing documents..."):
    docs, index, all_embs = load_and_index_with_progress()
st.success(f"Documents loaded successfully - total chunks: {len(docs)}")

# -----------------------------
# Session state (memory)
# -----------------------------
if "history_queue" not in st.session_state:
    st.session_state.history_queue = deque()

# -----------------------------
# Input interface
# -----------------------------
query = st.text_input("Enter your question:", placeholder="e.g. What is SAP HANA Cloud?")
send = st.button("Ask")

# -----------------------------
# Chat logic
# -----------------------------
if send and query:
    q_emb = get_embeddings_batch([query])[0]
    _, I = index.search(np.array([q_emb]), TOP_K)
    top_indices = I[0]
    top_docs = rerank(q_emb, all_embs, top_indices, docs, RERANK_TOP_M)

    combined_context = "\n".join(d["content"] for d in top_docs)
    full_history = "\n".join(list(st.session_state.history_queue))

    with st.spinner("Generating response..."):
        answer = chat(query, combined_context, full_history)

    sources = ", ".join(sorted(set(d["source"] for d in top_docs)))
    st.markdown("### Answer")
    st.write(answer)
    st.markdown(f"**Sources:** {sources}")

    # Update history
    st.session_state.history_queue, _ = update_history_queue(
        st.session_state.history_queue, query, answer
    )

# -----------------------------
# Display chat history
# -----------------------------
if st.session_state.history_queue:
    st.markdown("### Conversation History")
    for msg in list(st.session_state.history_queue):
        st.markdown(f"> {msg}")
