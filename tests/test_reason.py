import json
from pathlib import Path
import re

from handyllm import ChatPrompt, load_from
import responses


tests_dir = Path(__file__).parent
assets_dir = tests_dir / "assets"


mock_fetch_data = {
    "id": "27a49-098bc4-91d006",
    "object": "chat.completion",
    "created": 1755317006,
    "model": "deepseek-reasoner",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "A fractal gallery where infinitely intricate geometric sculptures bloom from pure equations.",
                "reasoning_content": 'Hmm, the user wants me to imagine a place where math and art intersect without involving physics. They specifically asked for a one-sentence answer with no explanation and minimal thinking. \n\nOkay, this is an interesting creative challenge. The user seems to want something poetic and abstract rather than factual. They\'re probably exploring philosophical or conceptual intersections rather than physical realities. \n\nI need to avoid any mention of physics, which rules out cosmic phenomena or natural patterns. Pure math and pure art... perhaps something about geometric forms or abstract concepts. \n\nThe instruction to "think as little as possible" suggests they want an instinctive, almost subconscious response. So I\'ll go with the first vivid image that comes to mind: an infinite gallery of impossible shapes. \n\nThis satisfies all criteria - it\'s purely mathematical (impossible shapes like Penrose triangles), purely artistic (gallery setting), and contains zero physics. The phrase "infinite gallery" also hints at the boundless nature of both disciplines. \n\nOne sentence, no explanations, minimal overthinking - done. The user seems to value brevity and creativity here, so this should work.',
            },
            "logprobs": None,
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 37,
        "completion_tokens": 241,
        "total_tokens": 278,
        "prompt_tokens_details": {"cached_tokens": 0},
        "completion_tokens_details": {"reasoning_tokens": 226},
        "prompt_cache_hit_tokens": 0,
        "prompt_cache_miss_tokens": 37,
    },
    "system_fingerprint": "fp_od0623_c3bca96",
}


mock_stream_data = [
    {
        "id": "537294aa-ac96f-b9ce0",
        "object": "chat.completion.chunk",
        "created": 1755316549,
        "model": "deepseek-reasoner",
        "system_fingerprint": "fp_653a9e_pr3bd0ca",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": "",
                },
                "logprobs": None,
                "finish_reason": None,
            }
        ],
    },
    {
        "id": "537294aa-ac96f-b9ce0",
        "object": "chat.completion.chunk",
        "created": 1755316549,
        "model": "deepseek-reasoner",
        "system_fingerprint": "fp_653a9e_pr3bd0ca",
        "choices": [
            {
                "index": 0,
                "delta": {"content": None, "reasoning_content": "H"},
                "logprobs": None,
                "finish_reason": None,
            }
        ],
    },
    {
        "id": "537294aa-ac96f-b9ce0",
        "object": "chat.completion.chunk",
        "created": 1755316549,
        "model": "deepseek-reasoner",
        "system_fingerprint": "fp_653a9e_pr3bd0ca",
        "choices": [
            {
                "index": 0,
                "delta": {"content": None, "reasoning_content": "mm"},
                "logprobs": None,
                "finish_reason": None,
            }
        ],
    },
    {
        "id": "537294aa-ac96f-b9ce0",
        "object": "chat.completion.chunk",
        "created": 1755316549,
        "model": "deepseek-reasoner",
        "system_fingerprint": "fp_653a9e_pr3bd0ca",
        "choices": [
            {
                "index": 0,
                "delta": {"content": None, "reasoning_content": "."},
                "logprobs": None,
                "finish_reason": None,
            }
        ],
    },
    {
        "id": "537294aa-ac96f-b9ce0",
        "object": "chat.completion.chunk",
        "created": 1755316549,
        "model": "deepseek-reasoner",
        "system_fingerprint": "fp_653a9e_pr3bd0ca",
        "choices": [
            {
                "index": 0,
                "delta": {"content": "The", "reasoning_content": None},
                "logprobs": None,
                "finish_reason": None,
            }
        ],
    },
    {
        "id": "537294aa-ac96f-b9ce0",
        "object": "chat.completion.chunk",
        "created": 1755316549,
        "model": "deepseek-reasoner",
        "system_fingerprint": "fp_653a9e_pr3bd0ca",
        "choices": [
            {
                "index": 0,
                "delta": {"content": " infinite", "reasoning_content": None},
                "logprobs": None,
                "finish_reason": None,
            }
        ],
    },
    {
        "id": "537294aa-ac96f-b9ce0",
        "object": "chat.completion.chunk",
        "created": 1755316549,
        "model": "deepseek-reasoner",
        "system_fingerprint": "fp_653a9e_pr3bd0ca",
        "choices": [
            {
                "index": 0,
                "delta": {"content": " complexity", "reasoning_content": None},
                "logprobs": None,
                "finish_reason": None,
            }
        ],
    },
    {
        "id": "537294aa-ac96f-b9ce0",
        "object": "chat.completion.chunk",
        "created": 1755316549,
        "model": "deepseek-reasoner",
        "system_fingerprint": "fp_653a9e_pr3bd0ca",
        "choices": [
            {
                "index": 0,
                "delta": {"content": ".", "reasoning_content": None},
                "logprobs": None,
                "finish_reason": None,
            }
        ],
    },
    {
        "id": "537294aa-ac96f-b9ce0",
        "object": "chat.completion.chunk",
        "created": 1755316549,
        "model": "deepseek-reasoner",
        "system_fingerprint": "fp_653a9e_pr3bd0ca",
        "choices": [
            {
                "index": 0,
                "delta": {"content": "", "reasoning_content": None},
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 37,
            "completion_tokens": 164,
            "total_tokens": 201,
            "prompt_tokens_details": {"cached_tokens": 0},
            "completion_tokens_details": {"reasoning_tokens": 147},
            "prompt_cache_hit_tokens": 0,
            "prompt_cache_miss_tokens": 37,
        },
    },
]

tmp = ["data: " + json.dumps(data) for data in mock_stream_data]
tmp.append("data: [DONE]")
stream_body = "\n".join(tmp)


@responses.activate
def test_reason_fetch():
    responses.add(responses.POST, url=re.compile(r".*"), json=mock_fetch_data)
    prompt = assets_dir / "reason.hprompt"
    prompt = load_from(prompt, cls=ChatPrompt)
    ret_prompt = prompt.run(api_key="fake-key")
    assert (
        ret_prompt.result_reasoning
        == 'Hmm, the user wants me to imagine a place where math and art intersect without involving physics. They specifically asked for a one-sentence answer with no explanation and minimal thinking. \n\nOkay, this is an interesting creative challenge. The user seems to want something poetic and abstract rather than factual. They\'re probably exploring philosophical or conceptual intersections rather than physical realities. \n\nI need to avoid any mention of physics, which rules out cosmic phenomena or natural patterns. Pure math and pure art... perhaps something about geometric forms or abstract concepts. \n\nThe instruction to "think as little as possible" suggests they want an instinctive, almost subconscious response. So I\'ll go with the first vivid image that comes to mind: an infinite gallery of impossible shapes. \n\nThis satisfies all criteria - it\'s purely mathematical (impossible shapes like Penrose triangles), purely artistic (gallery setting), and contains zero physics. The phrase "infinite gallery" also hints at the boundless nature of both disciplines. \n\nOne sentence, no explanations, minimal overthinking - done. The user seems to value brevity and creativity here, so this should work.'
    )
    assert (
        ret_prompt.result_str
        == "A fractal gallery where infinitely intricate geometric sculptures bloom from pure equations."
    )


@responses.activate
def test_reason_stream():
    responses.add(responses.POST, url=re.compile(r".*"), body=stream_body)
    prompt = assets_dir / "reason.hprompt"
    prompt = load_from(prompt, cls=ChatPrompt)
    ret_prompt = prompt.run(api_key="fake-key", stream=True)
    assert ret_prompt.result_reasoning == "Hmm."
    assert ret_prompt.result_str == "The infinite complexity."
