import streamlit as st

from preprocess import CustomDocumentLoader, split_text
from data import save_embeddings, fetch_model
from retrieval import retrieve
from scraper import fetch_sites

from langchain_core.prompts import ChatPromptTemplate

PROMPT_TEMPLATE = """
    Answer the question based only on the following context and in case the context is not adequate, clearly state so. Remember to explain any detail in the response since I do not know anything about the context:

    {context}

    ---

    Answer the question based on the above context and explain the answer clearly without alluding to the context in the response: {question}
    """


def main():
    st.set_page_config(page_title="LLM Summarizer", page_icon=":memo:")
    st.title(":memo: LLM Summarizer")       

    prompt = st.chat_input("Enter some query...")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt:
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner("Constructing context..."):
            context = []
            context.extend(fetch_sites(prompt))

        with st.spinner("Loading documents..."):
            loader = CustomDocumentLoader(context)

        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Processing query..."):
                documents = loader.lazy_load()
                chunks = split_text(documents, 1024, 256)
                save_embeddings(chunks)
                # db.add_documents(chunks)
                db = fetch_model()


                results = db.similarity_search_with_relevance_scores(prompt, k=3)
                if len(results) == 0 or results[0][1] < 0.5:
                    st.warning(f"Warning: Results may not be relevant.\nReason: The relevance scores for top 3 chunks are {[x[1] for x in results]}")

                context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
                prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
                query = prompt_template.format(context=context_text, question=prompt)

                # Collect response chunks
                response_chunks = retrieve(query, stream=True, model="meta-llama/Meta-Llama-3-8B-Instruct")
            msg = st.write_stream(response_chunks)
            print(msg)
            st.write("References:")
            for doc, _score in results:
                st.write(f" - Source: {doc.metadata.get('source', None)}")
                st.write(f"Link: {doc.metadata.get('link', None)}\n")

        # Add full assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": msg})

if __name__ == "__main__":
    main()