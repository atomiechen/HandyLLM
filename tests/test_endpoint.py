from handyllm import OpenAIAPI, EndpointManager, Endpoint

from dotenv import load_dotenv, find_dotenv
# load env parameters from file named .env
load_dotenv(find_dotenv())

import os

## EndpointManager acts like a list
endpoint_manager = EndpointManager()

endpoint_manager.add_endpoint_by_info(
    api_key=os.environ.get('OPENAI_API_KEY'),
)
endpoint2 = Endpoint(
    name='endpoint2',  # name is not required
    api_key=os.environ.get('OPENAI_API_KEY'),
)
endpoint_manager.append(endpoint2)

assert isinstance(endpoint_manager[0], Endpoint)
assert endpoint2 == endpoint_manager[1]
print(f"total endpoints: {len(endpoint_manager)}")

for endpoint in endpoint_manager:
    print(endpoint)
    # print(endpoint.get_api_info())  # WARNING: print endpoint info including api_key


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
    endpoint_manager=endpoint_manager
    )
print(response['choices'][0]['message']['content'])


print()
print("-----")


# ----- EXAMPLE 2 -----

response = OpenAIAPI.completions(
    model="text-davinci-002",
    prompt="count to 23 and stop: 1,2,3,",
    timeout=10,
    max_tokens=256,
    echo=True,  # Echo back the prompt in addition to the completion
    endpoint=endpoint2
)
print(response['choices'][0]['text'])
