from handyllm import OpenAIAPI

from dotenv import load_dotenv, find_dotenv
# load env parameters from file named .env
# API key is read from environment variable OPENAI_API_KEY
# organization is read from environment variable OPENAI_ORGANIZATION
load_dotenv(find_dotenv())

## or you can set these parameters in code
# OpenAIAPI.api_key = 'sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
# OpenAIAPI.organization = None

# ----- EXAMPLE 1 -----

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
    timeout=10,
    stream=True
    )

# you can use this to stream the response
for text in OpenAIAPI.stream_chat(response):
    print(text, end='')

# or you can use this to get the whole response
# for chunk in response:
#     if 'content' in chunk['choices'][0]['delta']:
#         print(chunk['choices'][0]['delta']['content'], end='')


print()
print("-----")


# ----- EXAMPLE 2 -----

response = OpenAIAPI.completions(
    model="text-davinci-002",
    prompt="count to 23 and stop: 1,2,3,",
    timeout=10,
    max_tokens=256,
    echo=True,  # Echo back the prompt in addition to the completion
    stream=True
)

# you can use this to stream the response
for text in OpenAIAPI.stream_completions(response):
    print(text, end='')

# or you can use this to get the whole response
# for chunk in response:
#     print(chunk['choices'][0]['text'], end='')
