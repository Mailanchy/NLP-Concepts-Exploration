import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))

try:
    print("Testing...")
    response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[{'role':'user', 'content':'സുഖമാണോ?'}]
    )
    print("Response from AI:", response.choices[0].message.content)
except Exception as e:
    print("Something went wrong")
    print(e)