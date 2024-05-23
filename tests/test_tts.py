from dotenv import load_dotenv, find_dotenv
# load env parameters from file
load_dotenv(find_dotenv('.env'))

import os
import asyncio

from handyllm import OpenAIClient
from handyllm import stream_to_file, astream_to_file

def sync_speech():
    with OpenAIClient() as client:
        response = client.audio_speech(
            model='tts-1',
            input="Hello, world! oh yes. This is a test. Sync speech no-stream version.",
            voice='alloy',
        ).call()
        with open('output-sync.mp3', 'wb') as f:
            f.write(response)

def sync_speech_stream():
    with OpenAIClient() as client:
        response = client.audio_speech(
            model='tts-1',
            input="Hello, world! oh yes. This is a test. Sync speech stream version.",
            voice='alloy',
            stream=True,
            chunk_size=1024,
        ).call()
        stream_to_file(response, 'output-sync-stream.mp3')

async def async_speech():
    async with OpenAIClient("async") as client:
        response = await client.audio_speech(
            model='tts-1',
            input="Hello, world! oh no. This is a test. Async speech no-stream version.",
            voice='alloy',
        ).acall()
        with open('output-async.mp3', 'wb') as f:
            f.write(response)

async def async_speech_stream():
    async with OpenAIClient("async") as client:
        response = await client.audio_speech(
            model='tts-1',
            input="Hello, world! oh no. This is a test. Async speech stream version.",
            voice='alloy',
            stream=True,
            chunk_size=1024,
        ).acall()
        await astream_to_file(response, 'output-async-stream.mp3')

def sync_speech_azure():
    with OpenAIClient(
        api_type='azure', 
        api_base=os.getenv("AZURE_OPENAI_ENDPOINT"), 
        api_key=os.getenv("AZURE_OPENAI_KEY"), 
        api_version=os.getenv("AZURE_OPENAI_API_VERSION")  # cannot be None if using Azure API.
        ) as client:
        # NOTE: Azure TTS API needs both deployment_id and model
        response = client.audio_speech(
            deployment_id='tts',
            model='tts-1',
            input="Hello, world! oh yes. This is a test. Sync speech no-stream version.",
            voice='alloy',
        ).call()
        with open('output-sync-azure.mp3', 'wb') as f:
            f.write(response)


if __name__ == '__main__':
    sync_speech()
    sync_speech_stream()
    asyncio.run(async_speech())
    asyncio.run(async_speech_stream())
    
    sync_speech_azure() # Azure TTS API

