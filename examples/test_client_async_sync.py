import asyncio
import time
import os
import logging

from handyllm import OpenAIClient
from handyllm import stream_chat, astream_chat

# log with time
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

from dotenv import load_dotenv, find_dotenv
# load env parameters from file
load_dotenv(find_dotenv('.env'))


def echo_kwargs(**kwargs):
    return kwargs

prompt = [{
    "role": "user",
    "content": "please tell me a joke"
    }]

kwargs = echo_kwargs(
    model="gpt-4-1106-preview",
    messages=prompt,
    temperature=0,
    max_tokens=256,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    timeout=15,
    stream=True,
)


def test_sync():
    # pass all parameters to the client constructor
    with OpenAIClient(
        "sync",
        api_type='azure',
        api_base=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        ) as client:
        response = client.chat(**kwargs).call()
        if kwargs.get('stream', False):
            for text in stream_chat(response):
                print(text, end='')
        else:
            print(response['choices'][0]['message']['content'])

async def test_async():
    async with OpenAIClient("async") as client:
        # set the parameters in the client
        client.api_type = 'azure'
        client.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
        client.api_key = os.getenv("AZURE_OPENAI_KEY")
        client.api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        response = await client.chat(**kwargs).acall()
        if kwargs.get('stream', False):
            async for text in astream_chat(response):
                print(text, end='')
        else:
            print(response['choices'][0]['message']['content'])

async def test_both():
    # client can be used as both sync and async
    async with OpenAIClient("both") as client:
        client.api_type = 'azure'
        client.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
        client.api_key = os.getenv("AZURE_OPENAI_KEY")
        client.api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        # async call
        response1 = await client.chat(**kwargs).acall()
        if kwargs.get('stream', False):
            async for text in astream_chat(response1):
                print(text, end='')
        else:
            print(response1['choices'][0]['message']['content'])
        # sync call
        response2 = client.chat(**kwargs).call()
        if kwargs.get('stream', False):
            for text in stream_chat(response2):
                print(text, end='')
        else:
            print(response2['choices'][0]['message']['content'])


if __name__ == "__main__":
    test_sync()
    time.sleep(5)
    asyncio.run(test_async())
    time.sleep(15)
    asyncio.run(test_both())

