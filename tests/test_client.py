import json
from pathlib import Path
import re
from handyllm import OpenAIClient
import responses


TEST_ROOT = Path(__file__).parent
ASSETS_ROOT = TEST_ROOT / "assets"


def test_client():
    client_key = 'client_key'
    client_base = 'https://api.example.com'
    client_org = 'client_org'
    client_api_version = '2024-01-01'
    client_api_type = 'azure'
    client_model_engine_map = {'model1': 'engine1', 'model2': 'engine2'}
    test_key_in_file1 = "test-key-in-file-1"
    test_key_in_file2 = "test-key-in-file-2"
    client = OpenAIClient(load_path=ASSETS_ROOT / "fake_credentials.yml")
    assert client.api_key == client_key
    assert client.api_base == client_base
    assert client.organization == client_org
    assert client.api_version == client_api_version
    assert client.api_type == client_api_type
    assert client.model_engine_map == client_model_engine_map
    assert client.endpoint_manager is not None
    assert client.endpoint_manager[0].api_key == test_key_in_file1
    assert client.endpoint_manager[1].api_key == test_key_in_file2

@responses.activate
def test_chat_fetch():
    mock_data = {
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
    
    responses.add(
        method=responses.POST,
        url=re.compile(r'.*'),
        json=mock_data,
    )
    
    with OpenAIClient('sync') as client:
        client.api_key = 'fake-key'
        response = client.chat(messages=[
            {'role': 'user', 'content': 'Hello!'},
        ]).fetch()
        print(response)
        assert response.choices[0].message["role"] == "assistant"
        assert response.usage.total_tokens == 21

@responses.activate
def test_chat_stream():
    mock_data = [
        {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-4o-mini", "system_fingerprint": "fp_44709d6fcb", "choices":[{"index":0,"delta":{"role":"assistant","content":""},"logprobs":None,"finish_reason":None}]},
        {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-4o-mini", "system_fingerprint": "fp_44709d6fcb", "choices":[{"index":0,"delta":{"content":"Hello"},"logprobs":None,"finish_reason":None}]},
        {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-4o-mini", "system_fingerprint": "fp_44709d6fcb", "choices":[{"index":0,"delta":{},"logprobs":None,"finish_reason":"stop"}]},
    ]
    tmp = ["data: " + json.dumps(data) for data in mock_data]
    tmp.append("data: [DONE]")
    body = "\n".join(tmp)

    responses.add(
        method=responses.POST,
        url=re.compile(r'.*'),
        body=body,
    )
    
    with OpenAIClient('sync') as client:
        client.api_key = 'fake-key'
        response = client.chat(messages=[
            {'role': 'user', 'content': 'Hello!'},
        ]).stream()
        result = ""
        for chunk in response:
            print(chunk)
            if 'role' in chunk.choices[0].delta:
                assert chunk.choices[0].delta['role'] == "assistant"
            if 'content' in chunk.choices[0].delta and chunk.choices[0].delta['content']:
                result += chunk.choices[0].delta['content']
        assert result == "Hello"

