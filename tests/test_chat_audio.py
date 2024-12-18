import base64
from pathlib import Path

from handyllm.hprompt import load_from, ChatPrompt


tests_dir = Path(__file__).parent


def test_chat_audio(tmp_path: Path):
    prompt_file = tests_dir / "assets" / "audio.hprompt"
    prompt = load_from(prompt_file, cls=ChatPrompt)

    # the audio file path should be a file uri (just for testing)
    assert (
        prompt.messages[-1]["content"][-1]["input_audio"]["data"] == "file://audio.mp3"
    )

    # create an example audio file
    audio_file = tmp_path / "audio.mp3"
    audio_file.touch()
    # write some content to the file
    fake_audio_data = b"abcdedfghijklmnopqrstuvwxyz"
    audio_file.write_bytes(fake_audio_data)

    # change the audio file path
    prompt.messages[-1]["content"][-1]["input_audio"]["data"] = (
        f"file://{audio_file.resolve()}"
    )

    evaled_prompt = prompt.eval()
    # the orginal prompt should not be changed
    assert (
        prompt.messages[-1]["content"][-1]["input_audio"]["data"]
        == f"file://{audio_file.resolve()}"
    )
    # the evaled prompt should have the audio data in base64
    assert evaled_prompt.messages[-1]["content"][-1]["input_audio"][
        "data"
    ] == base64.b64encode(fake_audio_data).decode("utf-8")
