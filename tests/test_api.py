from handyllm import OpenAIAPI

from dotenv import load_dotenv, find_dotenv
# load env parameters from file named .env
# API key is read from environment variable OPENAI_API_KEY
load_dotenv(find_dotenv())

## or you can set these parameters in code
OpenAIAPI.api_key = 'sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
# OpenAIAPI.organization = None

prompt = [{
    "role": "user",
    "content": "please tell me a joke"
    }]
response = OpenAIAPI.chat(
    model="gpt-3.5-turbo",
    messages=prompt,
    temperature=0.2,
    max_tokens=256,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    timeout=10
    )
print(response['choices'][0]['message']['content'])
