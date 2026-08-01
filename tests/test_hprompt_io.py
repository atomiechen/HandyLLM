from pathlib import Path

from handyllm import PromptConverter
from handyllm.hprompt import (
    HandyPrompt,
    load_from,
    dumps,
    dump_to,
    dump,
    ChatPrompt,
    CompletionsPrompt,
    RunConfig as RC,
)
from handyllm.utils import VM
from handyllm.types import Message


tests_dir = Path(__file__).parent
assets_dir = tests_dir / "assets"


def test_load_dump_no_frontmatter_prompt():
    prompt_file = assets_dir / "no_frontmatter.hprompt"
    prompt = load_from(prompt_file)
    # auto detect prompt type
    assert isinstance(prompt, ChatPrompt)
    # check no frontmatter in dump
    raw = prompt.dumps()
    assert "---" not in raw
    # specify prompt type
    prompt = CompletionsPrompt.load_from(prompt_file)
    assert isinstance(prompt, CompletionsPrompt)


def test_load_completions_prompt():
    prompt_file = assets_dir / "completions.hprompt"
    # auto detect prompt type
    prompt = HandyPrompt.load_from(prompt_file)
    assert isinstance(prompt, CompletionsPrompt)


def test_load_dump_chat_prompt(tmp_path):
    prompt_file = assets_dir / "chat.hprompt"
    prompt = load_from(prompt_file)
    assert isinstance(prompt, ChatPrompt)
    assert r"%extras%" in dumps(prompt)

    evaled_prompt = prompt.eval()
    out_path = tmp_path / "subdir" / "out.chat.hprompt"
    dump_to(evaled_prompt, out_path, mkdir=True)
    raw = out_path.read_text(encoding="utf-8")
    assert r"%extras%" not in raw
    assert "international" in raw

    with open(out_path, "w", encoding="utf-8") as f:
        dump(evaled_prompt, f)
    assert raw == out_path.read_text(encoding="utf-8")


def test_var_map():
    prompt_file = assets_dir / "chat.hprompt"
    prompt = load_from(prompt_file)
    evaled_prompt = prompt.eval(
        var_map=VM(
            context="We are having dinner now.",
        )
    )
    raw = evaled_prompt.dumps()
    assert "We are having dinner now." in raw

    evaled_prompt = prompt.eval(run_config=RC(var_map_path=assets_dir / "var_map.yml"))
    raw = evaled_prompt.dumps()
    assert r"%extras%" not in raw
    assert "Substituted text from YAML file." in raw
    assert "international" not in raw


def test_add_chat_prompt():
    prompt1 = ChatPrompt(
        messages=[
            {"role": "system", "content": "Hello there!"},
            {"role": "user", "content": "Hi!"},
        ]
    )
    prompt2 = ChatPrompt(
        messages=[
            {"role": "assistant", "content": "How are you?"},
            {"role": "user", "content": "I am fine."},
        ]
    )
    prompt = prompt1 + prompt2
    assert len(prompt.messages) == 4

    new: Message = {"role": "assistant", "content": "What can I do for you?"}
    prompt += new
    prompt += "I want to know more about the product."
    assert len(prompt.messages) == 6


def test_add_completions_prompt():
    prompt1 = CompletionsPrompt("This is a test.")
    prompt2 = CompletionsPrompt("This is indeed a test.")
    prompt = prompt1 + prompt2
    assert prompt.prompt == "This is a test.This is indeed a test."

    prompt += "Let's see if this works."
    assert (
        prompt.prompt == "This is a test.This is indeed a test.Let's see if this works."
    )


def test_extra_properties():
    prompt = assets_dir / "extra_properties.hprompt"
    loaded_prompt = load_from(prompt, cls=ChatPrompt)
    assert loaded_prompt.messages[0]["key1"] == "cont'en{\"'t}\n1"  # type: ignore
    assert loaded_prompt.messages[0]["key2"] == 'co"nt\'en"t2'  # type: ignore
    assert loaded_prompt.messages[0]["key3"] == "con{te{{nt}3"  # type: ignore


def test_extra_properties_json_round_trip():
    messages = [
        {
            "role": "assistant",
            "content": '{"answer": "ok"}',
            "refusal": None,
            "reasoning": "Used a tool and an array.",
            "reasoning_details": [
                {"type": "reasoning.text", "text": "Used the drawing tool safely."},
                {"type": "reasoning.text", "text": "Built an array of blocks."},
            ],
            "metadata": {"enabled": True, "score": 1.5},
        }
    ]
    converter = PromptConverter()
    raw = converter.msgs2raw(messages)  # type: ignore
    assert converter.raw2msgs(raw) == messages


def test_message_type_keywords_round_trip():
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "test", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "Hello"}],
        },
    ]
    converter = PromptConverter()
    raw = converter.msgs2raw(messages)  # type: ignore
    assert converter.raw2msgs(raw) == messages


def test_legacy_message_type_properties():
    converter = PromptConverter()
    tool_message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "test", "arguments": "{}"},
            }
        ],
    }
    array_message = {
        "role": "user",
        "content": [{"type": "text", "text": "Hello"}],
    }

    tool_raw = converter.msgs2raw([tool_message]).replace(  # type: ignore
        "{tool}", '{type="tool_calls"}'
    )
    array_raw = converter.msgs2raw([array_message]).replace(  # type: ignore
        "{array}", '{type="content_array"}'
    )

    assert converter.raw2msgs(tool_raw) == [tool_message]
    assert converter.raw2msgs(array_raw) == [array_message]
