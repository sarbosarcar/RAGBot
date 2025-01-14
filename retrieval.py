from openai import OpenAI
import os
import dotenv
from keybert import KeyBERT

dotenv.load_dotenv()
API_KEY = os.environ.get("HUGGINGFACEHUB_API_KEY")

model = KeyBERT('distilbert-base-nli-mean-tokens')

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
    # for chunk in result:
    #     print(chunk.choices[0].delta.content, end="")
    return result

def extract_keywords(text, limit : int = 4):
    keywords = model.extract_keywords(text, keyphrase_ngram_range=(1, limit), stop_words=None)
    return keywords