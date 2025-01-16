import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

import streamlit as st

# Save chunks to FAISS in memory
def save_embeddings(chunks):
    if not chunks:
        raise ValueError("Chunks not provided!")
    hf = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(chunks, hf)
    st.session_state.db = db
    print(f"Saved {len(chunks)} chunks to FAISS in memory")
    return db

# Load FAISS model from memory
def fetch_model():
    if "db" not in st.session_state or st.session_state.db is None:
        raise ValueError("FAISS index is not initialized. Please save embeddings first.")
    return st.session_state.db
