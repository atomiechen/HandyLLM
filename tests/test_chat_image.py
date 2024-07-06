from pathlib import Path

from handyllm.hprompt import load_from


tests_dir = Path(__file__).parent

def test_chat_image():
    prompt_file = tests_dir / 'assets' / 'image.hprompt'
    prompt = load_from(prompt_file)
    assert "data:image/jpeg;base64" not in prompt.dumps()
    
    evaled_prompt = prompt.eval()
    raw = evaled_prompt.dumps()
    assert "data:image/jpeg;base64" in raw
