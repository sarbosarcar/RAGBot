from langchain_core.prompts import ChatPromptTemplate

PROMPT_TEMPLATE = """
    Answer the question based only on the following context and in case the context is not adequate, clearly state so. Remember to explain any detail in the response since I do not know anything about the context:

    {context}

    ---

    Answer the question based on the above context and explain the answer clearly without alluding to the context in the response: {question}
    """

def prompt_answer(context_text : str, query : str, template : str = PROMPT_TEMPLATE):
    prompt_template = ChatPromptTemplate.from_template(template)
    return prompt_template.format(context=context_text, question=query)
