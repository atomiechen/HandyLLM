from pathlib import Path

from handyllm import hprompt
from handyllm.hprompt import ChatPrompt, CompletionsPrompt


tests_dir = Path(__file__).parent

def test_load_empty_prompt():
    prompt_file = tests_dir / 'assets' / 'empty.hprompt'
    prompt = hprompt.load_from(prompt_file)
    assert isinstance(prompt, ChatPrompt)

def test_load_completions_prompt():
    prompt_file = tests_dir / 'assets' / 'completions.hprompt'
    prompt = hprompt.load_from(prompt_file)
    assert isinstance(prompt, CompletionsPrompt)

def test_load_dump_chat_prompt(tmp_path):
    prompt_file = tests_dir / 'assets' / 'chat.hprompt'
    prompt = hprompt.load_from(prompt_file)
    assert isinstance(prompt, ChatPrompt)
    assert r'%extras%' in hprompt.dumps(prompt)
    evaled_prompt = prompt.eval()
    out_path = tmp_path / 'out.chat.hprompt'
    hprompt.dump_to(evaled_prompt, out_path)
    raw = out_path.read_text(encoding="utf-8")
    assert r'%extras%' not in raw and 'international' in raw
    with open(out_path, 'w', encoding='utf-8') as f:
        hprompt.dump(evaled_prompt, f)
    assert raw == out_path.read_text(encoding="utf-8")

