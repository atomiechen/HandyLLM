import base64
from pathlib import Path
from typing import cast

from handyllm.hprompt import load_from
from handyllm.types import FileContentPart


tests_dir = Path(__file__).parent


def test_chat_file(tmp_path: Path):
    prompt_file = tests_dir / "assets" / "file.hprompt"
    prompt = load_from(prompt_file)
    assert "data:application/pdf;base64" not in prompt.dumps()

    assert isinstance(prompt.messages[-1]["content"], list)

    # the pdf file path should be a file uri
    content_part = cast(FileContentPart, prompt.messages[-1]["content"][-1])
    assert "file_data" in content_part["file"]
    assert "filename" not in content_part["file"]
    assert content_part["file"]["file_data"] == "file://test.pdf"

    # create an example pdf file
    pdf_file = tmp_path / "test.pdf"
    pdf_file.touch()
    # write some content to the file
    fake_pdf_data = b"%PDF-1.4\n%Fake PDF content\n%%EOF"
    pdf_file.write_bytes(fake_pdf_data)

    # change the file path in the prompt
    content_part["file"]["file_data"] = f"file://{pdf_file.resolve()}"

    evaled_prompt = prompt.eval()
    # the orginal prompt should not be changed
    assert content_part["file"]["file_data"] == f"file://{pdf_file.resolve()}"

    assert isinstance(evaled_prompt.messages[-1]["content"], list)
    evaled_content_part = cast(
        FileContentPart, evaled_prompt.messages[-1]["content"][-1]
    )

    # the evaled prompt should have the file data in base64
    assert "file_data" in evaled_content_part["file"]
    assert evaled_content_part["file"][
        "file_data"
    ] == "data:application/pdf;base64," + base64.b64encode(fake_pdf_data).decode(
        "utf-8"
    )

    # the evaled prompt should have the extracted filename
    assert "filename" in evaled_content_part["file"]
    assert evaled_content_part["file"]["filename"] == "test.pdf"
