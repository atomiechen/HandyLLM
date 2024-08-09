import json
from pathlib import Path
import re
from handyllm import load_from, ChatPrompt, stream_chat_all, RunConfig
from pytest import CaptureFixture
import pytest
import responses
import respx


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

mock_stream_data = [
    {
        "id": "chatcmpl-xxxxxxxxxxxxxxxxxxxxx",
        "object": "chat.completion.chunk",
        "created": 1722879508,
        "model": "gpt-4o-2024-05-13",
        "system_fingerprint": "fp_c832e4513b",
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": None},
                "logprobs": None,
                "finish_reason": None,
            }
        ],
    },
    {
        "id": "chatcmpl-xxxxxxxxxxxxxxxxxxxxx",
        "object": "chat.completion.chunk",
        "created": 1722879508,
        "model": "gpt-4o-2024-05-13",
        "system_fingerprint": "fp_c832e4513b",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_xxxxxxxxxxxxxxxxxxxxxxxx",
                            "type": "function",
                            "function": {
                                "name": "get_current_weather",
                                "arguments": "",
                            },
                        }
                    ]
                },
                "logprobs": None,
                "finish_reason": None,
            }
        ],
    },
    {
        "id": "chatcmpl-xxxxxxxxxxxxxxxxxxxxx",
        "object": "chat.completion.chunk",
        "created": 1722879508,
        "model": "gpt-4o-2024-05-13",
        "system_fingerprint": "fp_c832e4513b",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [{"index": 0, "function": {"arguments": '{"lo'}}]
                },
                "logprobs": None,
                "finish_reason": None,
            }
        ],
    },
    {
        "id": "chatcmpl-xxxxxxxxxxxxxxxxxxxxx",
        "object": "chat.completion.chunk",
        "created": 1722879508,
        "model": "gpt-4o-2024-05-13",
        "system_fingerprint": "fp_c832e4513b",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [{"index": 0, "function": {"arguments": "catio"}}]
                },
                "logprobs": None,
                "finish_reason": None,
            }
        ],
    },
    {
        "id": "chatcmpl-xxxxxxxxxxxxxxxxxxxxx",
        "object": "chat.completion.chunk",
        "created": 1722879508,
        "model": "gpt-4o-2024-05-13",
        "system_fingerprint": "fp_c832e4513b",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [{"index": 0, "function": {"arguments": 'n": "S'}}]
                },
                "logprobs": None,
                "finish_reason": None,
            }
        ],
    },
    {
        "id": "chatcmpl-xxxxxxxxxxxxxxxxxxxxx",
        "object": "chat.completion.chunk",
        "created": 1722879508,
        "model": "gpt-4o-2024-05-13",
        "system_fingerprint": "fp_c832e4513b",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [{"index": 0, "function": {"arguments": "an F"}}]
                },
                "logprobs": None,
                "finish_reason": None,
            }
        ],
    },
    {
        "id": "chatcmpl-xxxxxxxxxxxxxxxxxxxxx",
        "object": "chat.completion.chunk",
        "created": 1722879508,
        "model": "gpt-4o-2024-05-13",
        "system_fingerprint": "fp_c832e4513b",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [{"index": 0, "function": {"arguments": "ranci"}}]
                },
                "logprobs": None,
                "finish_reason": None,
            }
        ],
    },
    {
        "id": "chatcmpl-xxxxxxxxxxxxxxxxxxxxx",
        "object": "chat.completion.chunk",
        "created": 1722879508,
        "model": "gpt-4o-2024-05-13",
        "system_fingerprint": "fp_c832e4513b",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [{"index": 0, "function": {"arguments": "sco, C"}}]
                },
                "logprobs": None,
                "finish_reason": None,
            }
        ],
    },
    {
        "id": "chatcmpl-xxxxxxxxxxxxxxxxxxxxx",
        "object": "chat.completion.chunk",
        "created": 1722879508,
        "model": "gpt-4o-2024-05-13",
        "system_fingerprint": "fp_c832e4513b",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [{"index": 0, "function": {"arguments": 'A", '}}]
                },
                "logprobs": None,
                "finish_reason": None,
            }
        ],
    },
    {
        "id": "chatcmpl-xxxxxxxxxxxxxxxxxxxxx",
        "object": "chat.completion.chunk",
        "created": 1722879508,
        "model": "gpt-4o-2024-05-13",
        "system_fingerprint": "fp_c832e4513b",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [{"index": 0, "function": {"arguments": '"unit'}}]
                },
                "logprobs": None,
                "finish_reason": None,
            }
        ],
    },
    {
        "id": "chatcmpl-xxxxxxxxxxxxxxxxxxxxx",
        "object": "chat.completion.chunk",
        "created": 1722879508,
        "model": "gpt-4o-2024-05-13",
        "system_fingerprint": "fp_c832e4513b",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [{"index": 0, "function": {"arguments": '": "fa'}}]
                },
                "logprobs": None,
                "finish_reason": None,
            }
        ],
    },
    {
        "id": "chatcmpl-xxxxxxxxxxxxxxxxxxxxx",
        "object": "chat.completion.chunk",
        "created": 1722879508,
        "model": "gpt-4o-2024-05-13",
        "system_fingerprint": "fp_c832e4513b",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [{"index": 0, "function": {"arguments": "hren"}}]
                },
                "logprobs": None,
                "finish_reason": None,
            }
        ],
    },
    {
        "id": "chatcmpl-xxxxxxxxxxxxxxxxxxxxx",
        "object": "chat.completion.chunk",
        "created": 1722879508,
        "model": "gpt-4o-2024-05-13",
        "system_fingerprint": "fp_c832e4513b",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [{"index": 0, "function": {"arguments": 'heit"'}}]
                },
                "logprobs": None,
                "finish_reason": None,
            }
        ],
    },
    {
        "id": "chatcmpl-xxxxxxxxxxxxxxxxxxxxx",
        "object": "chat.completion.chunk",
        "created": 1722879508,
        "model": "gpt-4o-2024-05-13",
        "system_fingerprint": "fp_c832e4513b",
        "choices": [
            {
                "index": 0,
                "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "}"}}]},
                "logprobs": None,
                "finish_reason": None,
            }
        ],
    },
    {
        "id": "chatcmpl-xxxxxxxxxxxxxxxxxxxxx",
        "object": "chat.completion.chunk",
        "created": 1722879508,
        "model": "gpt-4o-2024-05-13",
        "system_fingerprint": "fp_c832e4513b",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [
                        {
                            "index": 1,
                            "id": "call_yyyyyyyyyyyyyyyyyyyyyyyy",
                            "type": "function",
                            "function": {
                                "name": "get_current_weather",
                                "arguments": "",
                            },
                        }
                    ]
                },
                "logprobs": None,
                "finish_reason": None,
            }
        ],
    },
    {
        "id": "chatcmpl-xxxxxxxxxxxxxxxxxxxxx",
        "object": "chat.completion.chunk",
        "created": 1722879508,
        "model": "gpt-4o-2024-05-13",
        "system_fingerprint": "fp_c832e4513b",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [{"index": 1, "function": {"arguments": '{"lo'}}]
                },
                "logprobs": None,
                "finish_reason": None,
            }
        ],
    },
    {
        "id": "chatcmpl-xxxxxxxxxxxxxxxxxxxxx",
        "object": "chat.completion.chunk",
        "created": 1722879508,
        "model": "gpt-4o-2024-05-13",
        "system_fingerprint": "fp_c832e4513b",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [{"index": 1, "function": {"arguments": "catio"}}]
                },
                "logprobs": None,
                "finish_reason": None,
            }
        ],
    },
    {
        "id": "chatcmpl-xxxxxxxxxxxxxxxxxxxxx",
        "object": "chat.completion.chunk",
        "created": 1722879508,
        "model": "gpt-4o-2024-05-13",
        "system_fingerprint": "fp_c832e4513b",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [{"index": 1, "function": {"arguments": 'n": "N'}}]
                },
                "logprobs": None,
                "finish_reason": None,
            }
        ],
    },
    {
        "id": "chatcmpl-xxxxxxxxxxxxxxxxxxxxx",
        "object": "chat.completion.chunk",
        "created": 1722879508,
        "model": "gpt-4o-2024-05-13",
        "system_fingerprint": "fp_c832e4513b",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [{"index": 1, "function": {"arguments": "ew Y"}}]
                },
                "logprobs": None,
                "finish_reason": None,
            }
        ],
    },
    {
        "id": "chatcmpl-xxxxxxxxxxxxxxxxxxxxx",
        "object": "chat.completion.chunk",
        "created": 1722879508,
        "model": "gpt-4o-2024-05-13",
        "system_fingerprint": "fp_c832e4513b",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [{"index": 1, "function": {"arguments": "ork, "}}]
                },
                "logprobs": None,
                "finish_reason": None,
            }
        ],
    },
    {
        "id": "chatcmpl-xxxxxxxxxxxxxxxxxxxxx",
        "object": "chat.completion.chunk",
        "created": 1722879508,
        "model": "gpt-4o-2024-05-13",
        "system_fingerprint": "fp_c832e4513b",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [{"index": 1, "function": {"arguments": 'NY", "'}}]
                },
                "logprobs": None,
                "finish_reason": None,
            }
        ],
    },
    {
        "id": "chatcmpl-xxxxxxxxxxxxxxxxxxxxx",
        "object": "chat.completion.chunk",
        "created": 1722879508,
        "model": "gpt-4o-2024-05-13",
        "system_fingerprint": "fp_c832e4513b",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [{"index": 1, "function": {"arguments": "unit"}}]
                },
                "logprobs": None,
                "finish_reason": None,
            }
        ],
    },
    {
        "id": "chatcmpl-xxxxxxxxxxxxxxxxxxxxx",
        "object": "chat.completion.chunk",
        "created": 1722879508,
        "model": "gpt-4o-2024-05-13",
        "system_fingerprint": "fp_c832e4513b",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [{"index": 1, "function": {"arguments": '": "f'}}]
                },
                "logprobs": None,
                "finish_reason": None,
            }
        ],
    },
    {
        "id": "chatcmpl-xxxxxxxxxxxxxxxxxxxxx",
        "object": "chat.completion.chunk",
        "created": 1722879508,
        "model": "gpt-4o-2024-05-13",
        "system_fingerprint": "fp_c832e4513b",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [{"index": 1, "function": {"arguments": "ahrenh"}}]
                },
                "logprobs": None,
                "finish_reason": None,
            }
        ],
    },
    {
        "id": "chatcmpl-xxxxxxxxxxxxxxxxxxxxx",
        "object": "chat.completion.chunk",
        "created": 1722879508,
        "model": "gpt-4o-2024-05-13",
        "system_fingerprint": "fp_c832e4513b",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [{"index": 1, "function": {"arguments": 'eit"'}}]
                },
                "logprobs": None,
                "finish_reason": None,
            }
        ],
    },
    {
        "id": "chatcmpl-xxxxxxxxxxxxxxxxxxxxx",
        "object": "chat.completion.chunk",
        "created": 1722879508,
        "model": "gpt-4o-2024-05-13",
        "system_fingerprint": "fp_c832e4513b",
        "choices": [
            {
                "index": 0,
                "delta": {"tool_calls": [{"index": 1, "function": {"arguments": "}"}}]},
                "logprobs": None,
                "finish_reason": None,
            }
        ],
    },
    {
        "id": "chatcmpl-xxxxxxxxxxxxxxxxxxxxx",
        "object": "chat.completion.chunk",
        "created": 1722879508,
        "model": "gpt-4o-2024-05-13",
        "system_fingerprint": "fp_c832e4513b",
        "choices": [
            {"index": 0, "delta": {}, "logprobs": None, "finish_reason": "tool_calls"}
        ],
    },
]
tmp = ["data: " + json.dumps(data) for data in mock_stream_data]
tmp.append("data: [DONE]")
stream_body = "\n".join(tmp)


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


@responses.activate
def test_tool_call_stream(capsys: CaptureFixture[str]):
    responses.add(responses.POST, url=re.compile(r".*"), body=stream_body)
    prompt_file = tests_dir / "assets" / "chat_tool.hprompt"
    prompt = load_from(prompt_file, cls=ChatPrompt)
    response = prompt.stream(api_key="fake-key")
    tool_calls = []
    for chunk in stream_chat_all(response):
        role, content, tool_call = chunk
        tool_calls.append(tool_call)
        assert role == "assistant"
        assert content is None
    assert len(tool_calls) == 2
    assert tool_calls[0]["function"]["name"] == "get_current_weather"
    assert (
        tool_calls[0]["function"]["arguments"]
        == '{"location": "San Francisco, CA", "unit": "fahrenheit"}'
    )
    assert tool_calls[1]["function"]["name"] == "get_current_weather"
    assert (
        tool_calls[1]["function"]["arguments"]
        == '{"location": "New York, NY", "unit": "fahrenheit"}'
    )

    # make sure no debug prints
    captured = capsys.readouterr()
    assert captured.out == ""


@pytest.mark.asyncio
@respx.mock
@responses.activate
async def test_on_chunk_tool_call():
    responses.add(responses.POST, url=re.compile(r".*"), body=stream_body)
    respx.post(re.compile(r".*")).respond(text=stream_body)
    prompt_file = tests_dir / "assets" / "chat_tool.hprompt"
    prompt = load_from(prompt_file, cls=ChatPrompt)

    def on_chunk(role, content, tool_call):
        assert role == "assistant"
        assert content is None
        state.append(tool_call)

    async def aon_chunk(role, content, tool_call):
        assert role == "assistant"
        assert content is None
        state.append(tool_call)

    sync_run_config = RunConfig(on_chunk=on_chunk)
    async_run_config = RunConfig(on_chunk=aon_chunk)

    state = []
    prompt.run(run_config=sync_run_config, api_key="fake-key", stream=True)
    assert len(state) == 2

    state = []
    await prompt.arun(run_config=sync_run_config, api_key="fake-key", stream=True)
    assert len(state) == 2

    state = []
    await prompt.arun(run_config=async_run_config, api_key="fake-key", stream=True)
    assert len(state) == 2
