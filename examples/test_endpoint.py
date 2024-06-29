from handyllm import OpenAIClient, EndpointManager, Endpoint

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
    name='azure',  # name is not required
    api_type='azure', 
    api_base=os.getenv("AZURE_OPENAI_ENDPOINT"), 
    api_key=os.getenv("AZURE_OPENAI_KEY"), 
    api_version='2023-05-15',  # cannot be None if using Azure API.
    model_engine_map={  # needed if you want to use model alias
        'gpt-3.5-turbo': 'gpt-35-turbo'
    }
)
endpoint_manager.append(endpoint2)

assert isinstance(endpoint_manager[0], Endpoint)
assert endpoint2 == endpoint_manager[1]
print(f"total endpoints: {len(endpoint_manager)}")

for endpoint in endpoint_manager:
    print(endpoint)
    # print(endpoint.get_api_info())  # WARNING: print endpoint info including api_key


prompt = [{
    "role": "user",
    "content": "please tell me a joke"
    }]


def example1(client: OpenAIClient):
    # ----- EXAMPLE 1 -----

    try:
        response = client.chat(
            model="gpt-3.5-turbo",
            messages=prompt,
            temperature=0.2,
            max_tokens=256,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            timeout=10,
            endpoint_manager=endpoint_manager
            ).call()
        print(response['choices'][0]['message']['content'])
    except Exception as e:
        print(e)


def example2(client: OpenAIClient):
    # ----- EXAMPLE 2 -----

    try:
        response = client.chat(
            # deployment_id="initial_deployment",
            model="gpt-3.5-turbo",  # you can use model alias for Azure because model_engine_map is provided
            messages=prompt,
            timeout=10,
            max_tokens=256,
            endpoint=endpoint2
        ).call()
        print(response['choices'][0]['message']['content'])
    except Exception as e:
        print(e)


if __name__ == "__main__":
    with OpenAIClient() as client:
        example1(client)
        
        print()
        print("-----")
        
        example2(client)

