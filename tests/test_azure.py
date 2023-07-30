from handyllm import OpenAIAPI

from dotenv import load_dotenv, find_dotenv
# load env parameters from file
load_dotenv(find_dotenv('azure.env'))

import os
import json


OpenAIAPI.api_type = 'azure'
OpenAIAPI.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
OpenAIAPI.api_key = os.getenv("AZURE_OPENAI_KEY")
OpenAIAPI.api_version = '2023-05-15'  # can be None and default value will be used

# ----- EXAMPLE 1 -----

prompt = [{
    "role": "user",
    "content": "please tell me a joke"
    }]
response = OpenAIAPI.chat(
    # engine="gpt-35-turbo",
    deployment_id="initial_deployment",
    messages=prompt,
    temperature=0.2,
    max_tokens=256,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    timeout=10,
    )
print(response['choices'][0]['message']['content'])


print()
print("-----")


# ----- EXAMPLE 2 -----

response = OpenAIAPI.embeddings(
    engine="text-embedding-ada-002",
    input="I enjoy walking with my cute dog",
    timeout=10,
)
print(json.dumps(response, indent=2))

