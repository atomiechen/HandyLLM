from handyllm import OpenAIAPI

from dotenv import load_dotenv, find_dotenv
# load env parameters from file named .env
# API key is read from environment variable OPENAI_API_KEY
# organization is read from environment variable OPENAI_ORGANIZATION
load_dotenv(find_dotenv())

file_path = 'hello.m4a'

with open(file_path, "rb") as file_bin:
    response = OpenAIAPI.audio_transcriptions(
        file=file_bin,
        model='whisper-1',
        # timeout=10,
    )
print(response['text'])
