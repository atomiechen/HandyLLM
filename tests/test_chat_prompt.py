import json
from pathlib import Path
import re
from handyllm import (
    ChatPrompt, load_from, stream_chat_all, astream_chat_all,
    RunConfig
)
import pytest
import responses
import respx


tests_dir = Path(__file__).parent

mock_fetch_data = {
    "id": "chatcmpl-123",
    "object": "chat.completion",
    "created": 1677652288,
    "model": "gpt-4o-mini",
    "system_fingerprint": "fp_44709d6fcb",
    "choices": [{
        "index": 0,
        "message": {
            "role": "assistant",
            "content": "\n\nHello there, how may I assist you today?",
        },
        "logprobs": None,
        "finish_reason": "stop"
    }],
    "usage": {
        "prompt_tokens": 9,
        "completion_tokens": 12,
        "total_tokens": 21
    }
}

mock_stream_data = [
    {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-4o-mini", "system_fingerprint": "fp_44709d6fcb", "choices":[{"index":0,"delta":{"role":"assistant","content":""},"logprobs":None,"finish_reason":None}]},
    {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-4o-mini", "system_fingerprint": "fp_44709d6fcb", "choices":[{"index":0,"delta":{"content":"Hello"},"logprobs":None,"finish_reason":None}]},
    {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-4o-mini", "system_fingerprint": "fp_44709d6fcb", "choices":[{"index":0,"delta":{"content":" world!"},"logprobs":None,"finish_reason":None}]},
    {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-4o-mini", "system_fingerprint": "fp_44709d6fcb", "choices":[{"index":0,"delta":{},"logprobs":None,"finish_reason":"stop"}]},
]
tmp = ["data: " + json.dumps(data) for data in mock_stream_data]
tmp.append("data: [DONE]")
stream_body = "\n".join(tmp)


@responses.activate
def test_chat_fetch():
    responses.add(responses.POST, url=re.compile(r'.*'), json=mock_fetch_data)
    prompt_file = tests_dir / 'assets' / 'chat.hprompt'
    prompt = load_from(prompt_file, cls=ChatPrompt)
    response = prompt.fetch(api_key='fake-key')
    assert response.choices[0].message["role"] == "assistant"

@pytest.mark.asyncio
@respx.mock
async def test_async_chat_fetch():
    respx.post(re.compile(r'.*')).respond(json=mock_fetch_data)
    prompt_file = tests_dir / 'assets' / 'chat.hprompt'
    prompt = load_from(prompt_file, cls=ChatPrompt)
    response = await prompt.afetch(api_key='fake-key')
    assert response.choices[0].message["role"] == "assistant"

@responses.activate
def test_chat_stream():
    responses.add(responses.POST, url=re.compile(r'.*'), body=stream_body)
    prompt_file = tests_dir / 'assets' / 'chat.hprompt'
    prompt = load_from(prompt_file, cls=ChatPrompt)
    content = ""
    for role, text, tool_call in stream_chat_all(prompt.stream(api_key='fake-key')):
        print(role, text, tool_call)
        assert role == 'assistant'
        assert not tool_call
        assert text
        content += text
    assert content == "Hello world!"

@pytest.mark.asyncio
@respx.mock
async def test_async_chat_stream():
    respx.post(re.compile(r'.*')).respond(text=stream_body)
    prompt_file = tests_dir / 'assets' / 'chat.hprompt'
    prompt = load_from(prompt_file, cls=ChatPrompt)
    content = ""
    async for role, text, tool_call in astream_chat_all(prompt.astream(api_key='fake-key')):
        print(role, text, tool_call)
        assert role == 'assistant'
        assert not tool_call
        assert text
        content += text
    assert content == "Hello world!"

@responses.activate
def test_chat_run():
    responses.add(responses.POST, url=re.compile(r'.*'), json=mock_fetch_data)
    prompt_file = tests_dir / 'assets' / 'chat.hprompt'
    prompt = load_from(prompt_file, cls=ChatPrompt)
    result_prompt = prompt.run(api_key='fake-key')
    assert result_prompt.result_str == "\n\nHello there, how may I assist you today?"

    responses.replace(responses.POST, url=re.compile(r'.*'), body=stream_body)
    result_prompt = prompt.run(api_key='fake-key', stream=True)
    assert result_prompt.result_str == "Hello world!"

@pytest.mark.asyncio
@respx.mock
async def test_async_chat_run():
    respx.post(re.compile(r'.*')).respond(json=mock_fetch_data)
    prompt_file = tests_dir / 'assets' / 'chat.hprompt'
    prompt = load_from(prompt_file, cls=ChatPrompt)
    result_prompt = await prompt.arun(api_key='fake-key')
    assert result_prompt.result_str == "\n\nHello there, how may I assist you today?"

    respx.post(re.compile(r'.*')).respond(text=stream_body)
    result_prompt = await prompt.arun(api_key='fake-key', stream=True)
    assert result_prompt.result_str == "Hello world!"

@pytest.mark.asyncio
@respx.mock
@responses.activate
async def test_on_chunk_chat():
    responses.add(responses.POST, url=re.compile(r'.*'), body=stream_body)
    respx.post(re.compile(r'.*')).respond(text=stream_body)
    prompt_file = tests_dir / 'assets' / 'chat.hprompt'
    prompt = load_from(prompt_file, cls=ChatPrompt)
    
    def on_chunk(role, content, tool_call):
        state["role"] = role
        state['content'] += content
        state['tool_call'] = tool_call
    
    async def aon_chunk(role, content, tool_call):
        state["role"] = role
        state['content'] += content
        state['tool_call'] = tool_call
        
    sync_run_config = RunConfig(on_chunk=on_chunk)
    async_run_config = RunConfig(on_chunk=aon_chunk)
    
    state = {"content": ""}
    prompt.run(run_config=sync_run_config, api_key='fake-key', stream=True)
    assert state['role'] == 'assistant'
    assert state['content'] == "Hello world!"
    assert not state['tool_call']
    
    state = {"content": ""}
    await prompt.arun(run_config=sync_run_config, api_key='fake-key', stream=True)
    assert state['role'] == 'assistant'
    assert state['content'] == "Hello world!"
    assert not state['tool_call']

    state = {"content": ""}
    await prompt.arun(run_config=async_run_config, api_key='fake-key', stream=True)
    assert state['role'] == 'assistant'
    assert state['content'] == "Hello world!"
    assert not state['tool_call']

