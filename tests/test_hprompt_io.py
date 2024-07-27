from pathlib import Path

from handyllm.hprompt import (
    load_from, dumps, dump_to, dump, ChatPrompt, CompletionsPrompt, 
    RunConfig as RC
)
from handyllm.utils import VM


tests_dir = Path(__file__).parent

def test_load_dump_no_frontmatter_prompt():
    prompt_file = tests_dir / 'assets' / 'no_frontmatter.hprompt'
    prompt = load_from(prompt_file)
    # auto detect prompt type
    assert isinstance(prompt, ChatPrompt)
    # check no frontmatter in dump
    raw = prompt.dumps()
    assert '---' not in raw
    # specify prompt type
    prompt = load_from(prompt_file, cls=CompletionsPrompt)
    assert isinstance(prompt, CompletionsPrompt)

def test_load_completions_prompt():
    prompt_file = tests_dir / 'assets' / 'completions.hprompt'
    prompt = load_from(prompt_file)
    assert isinstance(prompt, CompletionsPrompt)

def test_load_dump_chat_prompt(tmp_path):
    prompt_file = tests_dir / 'assets' / 'chat.hprompt'
    prompt = load_from(prompt_file)
    assert isinstance(prompt, ChatPrompt)
    assert r'%extras%' in dumps(prompt)
    
    evaled_prompt = prompt.eval()
    out_path = tmp_path / 'subdir' /'out.chat.hprompt'
    dump_to(evaled_prompt, out_path, mkdir=True)
    raw = out_path.read_text(encoding="utf-8")
    assert r'%extras%' not in raw
    assert 'international' in raw
    
    with open(out_path, 'w', encoding='utf-8') as f:
        dump(evaled_prompt, f)
    assert raw == out_path.read_text(encoding="utf-8")

def test_var_map():
    prompt_file = tests_dir / 'assets' / 'chat.hprompt'
    prompt = load_from(prompt_file)
    evaled_prompt = prompt.eval(var_map=VM(
        context="We are having dinner now.",
    ))
    raw = evaled_prompt.dumps()
    assert 'We are having dinner now.' in raw
    
    evaled_prompt = prompt.eval(run_config=RC(
        var_map_path=tests_dir / 'assets' / 'var_map.yml'
    ))
    raw = evaled_prompt.dumps()
    assert r'%extras%' not in raw
    assert 'Substituted text from YAML file.' in raw
    assert 'international' not in raw

def test_add_chat_prompt():
    prompt1 = ChatPrompt(messages=[
        {'role': 'system', 'content': 'Hello there!'},
        {'role': 'user', 'content': 'Hi!'}
    ])
    prompt2 = ChatPrompt(messages=[
        {'role': 'assistant', 'content': 'How are you?'},
        {'role': 'user', 'content': 'I am fine.'}
    ])
    prompt = prompt1 + prompt2
    assert len(prompt.messages) == 4
    
    prompt += {'role': 'assistant', 'content': 'What can I do for you?'}
    prompt += "I want to know more about the product."
    assert len(prompt.messages) == 6

def test_add_completions_prompt():
    prompt1 = CompletionsPrompt("This is a test.")
    prompt2 = CompletionsPrompt("This is indeed a test.")
    prompt = prompt1 + prompt2
    assert prompt.prompt == "This is a test.This is indeed a test."

    prompt += "Let's see if this works."
    assert prompt.prompt == "This is a test.This is indeed a test.Let's see if this works."

