from openai import OpenAI
import os
import dotenv

dotenv.load_dotenv()
API_KEY = os.environ.get("HUGGINGFACEHUB_API_KEY")

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