from prompt import prompt_answer
from retrieval import retrieve

PROMPT_TEMPLATE = """
    Analyse the following context and determine if it is relevant to the question. In case it is not relevant, return 0, otherwise return 1:

    {context}

    ---

    Return only the number 0 or 1 based on the context and the following question: {question}
    """

def check_relevance(query : str, context : str):
    prompt = prompt_answer(context, query, PROMPT_TEMPLATE)
    response = retrieve(prompt, stream=False, model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")
    if "0" in response.choices[0].message.content:
        return False
    else:
        return True
