from handyllm import OpenAIAPI

import json
from dotenv import load_dotenv, find_dotenv
# load env parameters from file named .env
# API key is read from environment variable OPENAI_API_KEY
# organization is read from environment variable OPENAI_ORGANIZATION
load_dotenv(find_dotenv())

## or you can set these parameters in code
# OpenAIAPI.api_key = 'sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
# OpenAIAPI.organization = None

## get all models
response = OpenAIAPI.models_list(
    timeout=10,
)
print(json.dumps(response, indent=2))

## retrieve a specific model
response = OpenAIAPI.models_retrieve(
    timeout=10,
    model="text-embedding-ada-002",
)
print(json.dumps(response, indent=2))
