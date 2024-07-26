from pathlib import Path
import re
from handyllm import ChatPrompt, load_from
import responses


tests_dir = Path(__file__).parent

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
    
    prompt_file = tests_dir / 'assets' / 'chat.hprompt'
    prompt = load_from(prompt_file, cls=ChatPrompt)
    response = prompt.fetch(api_key='fake-key')
    assert response.choices[0].message["role"] == "assistant"

