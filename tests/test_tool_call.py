from pathlib import Path
import re
from handyllm import load_from, ChatPrompt
import responses


tests_dir = Path(__file__).parent

mock_fetch_data = {
    "id": "chatcmpl-xxxxxxxxxxxxxxxxxxxxxx",
    "object": "chat.completion",
    "created": 1722818900,
    "model": "gpt-4o-2024-05-13",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_xxxxxxxxxxxxxxxxxxxxxxxx",
                        "type": "function",
                        "function": {
                            "name": "get_current_weather",
                            "arguments": '{"location": "San Francisco, CA", "unit": "celsius"}',
                        },
                    },
                    {
                        "id": "call_yyyyyyyyyyyyyyyyyyyyyyyy",
                        "type": "function",
                        "function": {
                            "name": "get_current_weather",
                            "arguments": '{"location": "New York, NY", "unit": "celsius"}',
                        },
                    },
                ],
            },
            "logprobs": None,
            "finish_reason": "tool_calls",
        }
    ],
    "usage": {"prompt_tokens": 89, "completion_tokens": 62, "total_tokens": 151},
    "system_fingerprint": "fp_3cd8b62c3b",
}


@responses.activate
def test_tool_call():
    responses.add(responses.POST, url=re.compile(r".*"), json=mock_fetch_data)
    prompt_file = tests_dir / "assets" / "chat_tool.hprompt"
    prompt = load_from(prompt_file, cls=ChatPrompt)
    response = prompt.fetch(api_key="fake-key")
    assert "tool_calls" in response.choices[0].message
    assert (
        response.choices[0].message["tool_calls"][0]["function"]["name"]
        == "get_current_weather"
    )
    assert (
        response.choices[0].message["tool_calls"][0]["function"]["arguments"]
        == '{"location": "San Francisco, CA", "unit": "celsius"}'
    )
    assert (
        response.choices[0].message["tool_calls"][1]["function"]["name"]
        == "get_current_weather"
    )
    assert (
        response.choices[0].message["tool_calls"][1]["function"]["arguments"]
        == '{"location": "New York, NY", "unit": "celsius"}'
    )
