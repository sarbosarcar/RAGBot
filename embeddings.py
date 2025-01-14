from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import os
from langchain_huggingface import HuggingFaceEmbeddings


def get_embeddings():
    # Load the pre-trained embedding model from SentenceTransformers
    model = SentenceTransformer("all-MiniLM-L6-v2")  # Compact and fast open-source model

    # Get embeddings for words
    word1 = "apple"
    word2 = "orange"

    embedding1 = model.encode(word1, convert_to_tensor=True)
    embedding2 = model.encode(word2, convert_to_tensor=True)

    # print(f"Vector for '{word1}': {embedding1}")
    # print(f"Vector for '{word2}': {embedding2}")
    # print(f"Vector length: {len(embedding1)}")

    # # Compare vectors using cosine similarity
    # similarity = 1 - cosine(embedding1, embedding2)  # Higher is more similar
    # print(f"Similarity between '{word1}' and '{word2}': {similarity:.4f}")

    return embedding1, embedding2

def embedding_function():
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}
    hf = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    return hf