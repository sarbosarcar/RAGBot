import streamlit as st

from preprocess import CustomDocumentLoader, split_text
from prompt import prompt_answer
from postprocess import check_relevance
from data import save_embeddings, fetch_model
from retrieval import retrieve, retrieve_sonar
from scraper import fetch_sites

def main():
    st.set_page_config(page_title="RAGBot", page_icon=":memo:")
    st.title(":memo: RAGBot")       

    prompt = st.chat_input("Enter some query...")
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that can scrape the web for responses to user queries. You should provide relevant sources for each answer and clearly enunciate the answer in a clear and concise manner. In case a question is not relevant or not answerable, please state that you cannot answer."
                ),
            },
        ]

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"]!="system":
                st.markdown(message["content"])

    if prompt:
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # with st.spinner("Constructing context..."):
        #     context = []
        #     context.extend(fetch_sites(prompt))

        # with st.spinner("Loading documents..."):
        #     loader = CustomDocumentLoader(context)

        # Generate assistant response
        
        with st.chat_message("assistant"):
            with st.spinner("Processing query..."):
                print(st.session_state.messages)
                response = retrieve_sonar(st.session_state.messages, prompt)
                
            #     documents = loader.lazy_load()
            #     chunks = split_text(documents, 1024, 256)
            #     save_embeddings(chunks)
            #     # db.add_documents(chunks)
            #     db = fetch_model()


            #     results = db.similarity_search_with_relevance_scores(prompt, k=3)
            #     if len(results) == 0 or results[0][1] < 0.5:
            #         st.warning(f"Warning: Results may not be relevant.\nReason: The relevance scores for top 3 chunks are {[x[1] for x in results]}")

            #     context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

            #     query = prompt_answer(context_text, prompt)

            #     relevance = check_relevance(prompt, context_text)
            #     print(f"relevance: {relevance}")
            #     # Collect response chunks
            #     if relevance:
            #         response_chunks = retrieve(query, stream=True, model="meta-llama/Meta-Llama-3-8B-Instruct")
            #         msg = st.write_stream(response_chunks)
            #     else:
            #         msg = st.warning("The context is not relevant to the question.")
                
            # st.write("References:")
            
            # for doc, _score in results:
            #     st.write(f" - Source: {doc.metadata.get('source', None)}")
            #     st.write(f"Link: {doc.metadata.get('link', None)}\n")
            # """
            reply = st.write_stream(response)

        # Add full assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": reply})

if __name__ == "__main__":
    main()