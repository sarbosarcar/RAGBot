from openai import OpenAI
import os
import dotenv

dotenv.load_dotenv()
API_KEY = os.environ.get("HUGGINGFACEHUB_API_KEY")
PPLX_API_KEY = os.environ.get("PERPLEXITY_API_KEY")

def retrieve(prompt, stream=True, model="HuggingFaceH4/zephyr-7b-beta"):
    client = OpenAI(
        base_url="https://api-inference.huggingface.co/v1/",
        api_key=f"{API_KEY}"
    )
    messages = [
        {
            "role": "user",
            "content": f"{prompt}"
        }
    ]
    result = client.chat.completions.create(
        model=model, 
        messages=messages, 
        max_tokens=1024,
        stream=stream
    )
    return result

def retrieve_sonar(message, prompt, stream=True):
    client = OpenAI(api_key=PPLX_API_KEY, base_url="https://api.perplexity.ai")
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that can scrape the web for responses to user queries. You should provide relevant sources for each answer and clearly enunciate the answer in a clear and concise manner. In case a question is not relevant or not answerable, please state that you cannot answer."
            ),
        },
        {   
            "role": "user",
            "content": (
                f"{prompt}"
            ),
        },
    ]

    # chat completion without streaming
    response = client.chat.completions.create(
        model="sonar",
        messages=message,
        stream=stream
    )
    return response
