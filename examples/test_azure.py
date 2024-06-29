from handyllm import OpenAIClient, utils

from dotenv import load_dotenv, find_dotenv
# load env parameters from file
load_dotenv(find_dotenv())

import os
import json


def example_chat(client: OpenAIClient):
    # ----- EXAMPLE 1 -----

    prompt = [{
        "role": "user",
        "content": "please tell me a joke"
        }]
    response = client.chat(
        # this is the engine (i.e. deployment_id) parameter for Azure OpenAI API
        engine="gpt-35-turbo",
        
        # # OR: you can use model parameter and specify the model_engine_map
        # # this is most useful for EndpointManager to unify API calls
        # model="gpt-3.5-turbo",
        # model_engine_map={"gpt-3.5-turbo": "gpt-35-turbo"},
        
        messages=prompt,
        temperature=0.2,
        max_tokens=256,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        timeout=10,
        ).call()
    print(response['choices'][0]['message']['content'])



def example_embeddings(client: OpenAIClient):
    # ----- EXAMPLE 2 -----

    response = client.embeddings(
        engine="text-embedding-ada-002",
        input="I enjoy walking with my cute dog",
        timeout=10,
    ).call()
    print(json.dumps(response, indent=2))


def example_images_generations(client: OpenAIClient):
    # ----- EXAMPLE 3 -----

    response = client.images_generations(
        api_version='2023-06-01-preview',
        prompt="A panda, synthwave style, digital painting",
        n=1,
        size="256x256",
    ).call()
    print(json.dumps(response, indent=2))
    download_url = response['data'][0]['url']
    file_path = utils.download_binary(download_url)
    print(f"generated image: {file_path}")


if __name__ == "__main__":
    with OpenAIClient(
        api_type='azure', 
        api_base=os.getenv("AZURE_OPENAI_ENDPOINT"), 
        api_key=os.getenv("AZURE_OPENAI_KEY"), 
        api_version=os.getenv("AZURE_OPENAI_API_VERSION")  # cannot be None if using Azure API.
        ) as client:
        example_chat(client)
        
        print()
        print("-----")
        
        example_embeddings(client)
        
        print()
        print("-----")
        
        example_images_generations(client)

