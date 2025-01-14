import os
import shutil

from langchain_chroma import Chroma
from chromadb.utils import embedding_functions
from langchain_huggingface import HuggingFaceEmbeddings

from embeddings import embedding_function

from preprocess import clear_dir

CHROMA_PATH = "chroma"

def save_embeddings(chunks):
    # clear_dir(CHROMA_PATH)

    hf = embedding_function()

    # db = Chroma.from_documents(
    #     chunks, hf, persist_directory=CHROMA_PATH
    # )

    db = Chroma.from_documents(
        chunks, hf
    )

    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}")

    return db

def fetch_model():
    ef = embedding_function()
    # db = Chroma(persist_directory=CHROMA_PATH, embedding_function=ef)
    db = Chroma(embedding_function=ef)
    return db

def del_db():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)