import json
from pathlib import Path
import re
from handyllm import ChatPrompt, load_from
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
body = "\n".join(tmp)


@responses.activate
def test_chat_fetch():
    responses.add(
        method=responses.POST,
        url=re.compile(r'.*'),
        json=mock_fetch_data,
    )
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
    responses.add(
        method=responses.POST,
        url=re.compile(r'.*'),
        body=body,
    )
    
    prompt_file = tests_dir / 'assets' / 'chat.hprompt'
    prompt = load_from(prompt_file, cls=ChatPrompt)
    content = ""
    for role, text, tool_call in prompt.stream(api_key='fake-key'):
        print(role, text, tool_call)
        assert role == 'assistant'
        assert not tool_call
        assert text
        content += text
    assert content == "Hello world!"

@pytest.mark.asyncio
@respx.mock
async def test_async_chat_stream():
    respx.post(re.compile(r'.*')).respond(text=body)
    
    prompt_file = tests_dir / 'assets' / 'chat.hprompt'
    prompt = load_from(prompt_file, cls=ChatPrompt)
    content = ""
    async for role, text, tool_call in prompt.astream(api_key='fake-key'):
        print(role, text, tool_call)
        assert role == 'assistant'
        assert not tool_call
        assert text
        content += text
    assert content == "Hello world!"

