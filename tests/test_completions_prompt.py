import json
from pathlib import Path
import re

from handyllm import (
    CompletionsPrompt, load_from, stream_completions, astream_completions,
    RunConfig,
)
import pytest
import responses
import respx


tests_dir = Path(__file__).parent

mock_fetch_data = {
  "id": "cmpl-uqkvlQyYK7bGYrRHQ0eXlWi7",
  "object": "text_completion",
  "created": 1589478378,
  "model": "gpt-3.5-turbo-instruct",
  "system_fingerprint": "fp_44709d6fcb",
  "choices": [
    {
      "text": "\n\nThis is indeed a test",
      "index": 0,
      "logprobs": None,
      "finish_reason": "length"
    }
  ],
  "usage": {
    "prompt_tokens": 5,
    "completion_tokens": 7,
    "total_tokens": 12
  }
}

mock_stream_data = [
    {"choices":[{"text":""}]},
    {"choices":[{"text":"This"}]},
    {"choices":[{"text":" is indeed a test"}]},
]
tmp = ["data: " + json.dumps(data) for data in mock_stream_data]
tmp.append("data: [DONE]")
stream_body = "\n".join(tmp)


@responses.activate
def test_completions_fetch():
    responses.add(responses.POST, url=re.compile(r'.*'), json=mock_fetch_data)
    prompt_file = tests_dir / 'assets' / 'completions.hprompt'
    prompt = load_from(prompt_file, cls=CompletionsPrompt)
    response = prompt.fetch(api_key='fake-key')
    assert response.choices[0].text == "\n\nThis is indeed a test"

@pytest.mark.asyncio
@respx.mock
async def test_async_completions_fetch():
    respx.post(re.compile(r'.*')).respond(json=mock_fetch_data)
    prompt_file = tests_dir / 'assets' / 'completions.hprompt'
    prompt = load_from(prompt_file, cls=CompletionsPrompt)
    response = await prompt.afetch(api_key='fake-key')
    assert response.choices[0].text == "\n\nThis is indeed a test"

@responses.activate
def test_completions_stream():
    responses.add(responses.POST, url=re.compile(r'.*'), body=stream_body)
    prompt_file = tests_dir / 'assets' / 'completions.hprompt'
    prompt = load_from(prompt_file, cls=CompletionsPrompt)
    response = prompt.stream(api_key='fake-key')
    content = ""
    for text in stream_completions(response):
        content += text
    assert content == "This is indeed a test"

@pytest.mark.asyncio
@respx.mock
async def test_async_completions_stream():
    respx.post(re.compile(r'.*')).respond(text=stream_body)
    prompt_file = tests_dir / 'assets' / 'completions.hprompt'
    prompt = load_from(prompt_file, cls=CompletionsPrompt)
    response = prompt.astream(api_key='fake-key')
    content = ""
    async for text in astream_completions(response):
        content += text
    assert content == "This is indeed a test"

@responses.activate
def test_completions_run():
    responses.add(responses.POST, url=re.compile(r'.*'), json=mock_fetch_data)
    prompt_file = tests_dir / 'assets' / 'completions.hprompt'
    prompt = load_from(prompt_file, cls=CompletionsPrompt)
    result_prompt = prompt.run(api_key='fake-key')
    assert result_prompt.result_str == "\n\nThis is indeed a test"
    
    responses.replace(responses.POST, url=re.compile(r'.*'), body=stream_body)
    result_prompt = prompt.run(api_key='fake-key', stream=True)
    assert result_prompt.result_str == "This is indeed a test"

@pytest.mark.asyncio
@respx.mock
async def test_async_completions_run():
    respx.post(re.compile(r'.*')).respond(json=mock_fetch_data)
    prompt_file = tests_dir / 'assets' / 'completions.hprompt'
    prompt = load_from(prompt_file, cls=CompletionsPrompt)
    result_prompt = await prompt.arun(api_key='fake-key')
    assert result_prompt.result_str == "\n\nThis is indeed a test"
    
    respx.post(re.compile(r'.*')).respond(text=stream_body)
    result_prompt = await prompt.arun(api_key='fake-key', stream=True)
    assert result_prompt.result_str == "This is indeed a test"

@pytest.mark.asyncio
@respx.mock
@responses.activate
async def test_on_chunk_completions():
    responses.add(responses.POST, url=re.compile(r'.*'), body=stream_body)
    respx.post(re.compile(r'.*')).respond(text=stream_body)
    prompt_file = tests_dir / 'assets' / 'completions.hprompt'
    prompt = load_from(prompt_file, cls=CompletionsPrompt)
    
    def on_chunk(data):
        state['content'] += data
    
    async def aon_chunk(data):
        state['content'] += data
    
    sync_run_config = RunConfig(on_chunk=on_chunk)
    async_run_config = RunConfig(on_chunk=aon_chunk)
    
    state = {"content": ""}
    prompt.run(run_config=sync_run_config, api_key='fake-key', stream=True)
    assert state["content"] == "This is indeed a test"
    
    state = {"content": ""}
    await prompt.arun(run_config=sync_run_config, api_key='fake-key', stream=True)
    assert state["content"] == "This is indeed a test"
    
    state = {"content": ""}
    await prompt.arun(run_config=async_run_config, api_key='fake-key', stream=True)
    assert state["content"] == "This is indeed a test"

