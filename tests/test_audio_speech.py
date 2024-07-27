from pathlib import Path
import re

from handyllm import OpenAIClient
from handyllm import stream_to_file, astream_to_file
import pytest
import responses
import respx


stream_body = b'abcdedfghijklmnopqrstuvwxyz'

@responses.activate
def test_sync_speech():
    responses.add(responses.POST, url=re.compile(r'.*'), body=stream_body)
    with OpenAIClient() as client:
        response = client.audio_speech(api_key='fake-key').fetch()
    assert response == stream_body

@responses.activate
def test_sync_speech_stream(tmp_path: Path):
    responses.add(responses.POST, url=re.compile(r'.*'), body=stream_body)
    file_path = tmp_path / 'output-sync-stream.mp3'
    with OpenAIClient() as client:
        response = client.audio_speech(api_key='fake-key').stream()
        stream_to_file(response, file_path)
    assert file_path.read_bytes() == stream_body

@pytest.mark.asyncio
@respx.mock
async def test_async_speech():
    respx.post(re.compile(r'.*')).respond(content=stream_body)
    async with OpenAIClient("async") as client:
        response = await client.audio_speech(api_key='fake-key').afetch()
    assert response == stream_body

@pytest.mark.asyncio
@respx.mock
async def test_async_speech_stream(tmp_path: Path):
    respx.post(re.compile(r'.*')).respond(content=stream_body)
    file_path = tmp_path / 'output-async-stream.mp3'
    async with OpenAIClient("async") as client:
        response = await client.audio_speech(api_key='fake-key').astream()
        await astream_to_file(response, file_path)
    assert file_path.read_bytes() == stream_body

