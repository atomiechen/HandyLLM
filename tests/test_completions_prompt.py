from pathlib import Path
import re

from handyllm import CompletionsPrompt, load_from
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
def test_completions_run():
    responses.add(responses.POST, url=re.compile(r'.*'), json=mock_fetch_data)
    prompt_file = tests_dir / 'assets' / 'completions.hprompt'
    prompt = load_from(prompt_file, cls=CompletionsPrompt)
    result_prompt = prompt.run(api_key='fake-key')
    assert result_prompt.result_str == "\n\nThis is indeed a test"

@pytest.mark.asyncio
@respx.mock
async def test_async_completions_run():
    respx.post(re.compile(r'.*')).respond(json=mock_fetch_data)
    prompt_file = tests_dir / 'assets' / 'completions.hprompt'
    prompt = load_from(prompt_file, cls=CompletionsPrompt)
    result_prompt = await prompt.arun(api_key='fake-key')
    assert result_prompt.result_str == "\n\nThis is indeed a test"

