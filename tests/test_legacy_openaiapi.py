import re
from handyllm import OpenAIAPI
import pytest
import responses


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

@responses.activate
def test_chat():
    responses.add(responses.POST, url=re.compile(r'.*'), json=mock_fetch_data)
    OpenAIAPI.api_key = 'fake-key'
    OpenAIAPI.api_base = 'https://api.fake.com'
    OpenAIAPI.api_type = 'openai'
    OpenAIAPI.api_version = '2021-08-04'
    OpenAIAPI.model_engine_map = {}
    response = OpenAIAPI.chat(messages=[
        {'role': 'user', 'content': 'Hello world!'}
    ])
    assert isinstance(response, dict)
    assert response['choices'][0]['message']['role'] == "assistant"
    assert response['choices'][0]['message']['content'] == "\n\nHello there, how may I assist you today?"

def test_unknown_api():
    with pytest.raises(AttributeError) as excinfo:
        OpenAIAPI.unknown_api()
    assert "'OpenAIAPI' has no attribute 'unknown_api'" == str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        OpenAIAPI.load_from()
    assert "load_from is not an API method" == str(excinfo.value)

