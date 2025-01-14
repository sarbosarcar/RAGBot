from preprocess import load_documents, split_text
# from embeddings import get_embeddings
from data import save_embeddings, fetch_model, CHROMA_PATH
from retrieval import retrieve

import argparse

# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate




PROMPT_TEMPLATE = """
Answer the question based only on the following context and in case the context is not adequate, clearly state so.:

{context}

---

Answer the question based on the above context: {question}
"""


def main():
    # Create CLI.
    documents = load_documents()
    chunks = split_text(documents, 200, 100)

    save_embeddings(chunks)                

    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    # Prepare the DB.
    db = fetch_model()

    # Search the DB.
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    if len(results) == 0 or results[0][1] < 0.5:
        print(f"Exception: Unable to find relevant results.")
        print(f"Reason: The relevance scores for top 3 chunks are {[x[1] for x in results]}")

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    # print(prompt)

    response = retrieve(prompt, stream=False)

    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response.choices[0].message.content}\nSources: {sources}"
    print(formatted_response)


if __name__ == "__main__":
    main()