import base64
import copy
import mimetypes
from pathlib import Path
from typing import (
    IO,
    AsyncGenerator,
    AsyncIterable,
    Generator,
    Iterable,
    List,
    Optional,
    TypeVar,
    cast,
)
from urllib.parse import urlparse
from urllib.request import url2pathname
import os
import time

from .types import (
    AudioContentPart,
    ChatChunkUnified,
    CustomToolCallDelta,
    FileContentPart,
    FileObject,
    FunctionToolCallDelta,
    ImageContentPart,
    PathType,
    ShortChatChunk,
    ChatChunk,
    CompletionsChunk,
    TextContentPart,
    ToolCallDelta,
)


YieldType = TypeVar("YieldType")


def get_filename_from_url(download_url):
    # Parse the URL.
    parsed_url = urlparse(download_url)
    # The last part of the path is usually the filename.
    filename = os.path.basename(parsed_url.path)
    return filename


def download_binary(download_url, file_path=None, dir="."):
    import requests

    response = requests.get(download_url, allow_redirects=True)
    if file_path is None:
        filename = get_filename_from_url(download_url)
        if filename == "" or filename is None:
            filename = "download_" + time.strftime("%Y%m%d_%H%M%S")
        file_path = os.path.abspath(os.path.join(dir, filename))
    # Open the file in binary mode and write to it.
    with open(file_path, "wb") as file:
        file.write(response.content)
    return file_path


def trans_stream_chat(
    consumer: Generator[YieldType, ShortChatChunk, None],
) -> Generator[Optional[YieldType], Optional[ChatChunk], None]:
    next(consumer)  # prime the generator
    role = ""
    tool_call = None
    ret = None
    try:
        while True:
            data = yield ret
            if data is None:
                break
            ret = None
            try:
                message = data["choices"][0]["delta"]
                if "role" in message:
                    role = cast(str, message["role"])
                content = cast(Optional[str], message.get("content"))
                reasoning_content = cast(
                    Optional[str], message.get("reasoning_content")
                )
                tool_calls = cast(
                    Optional[List[ToolCallDelta]], message.get("tool_calls")
                )
                if tool_calls:
                    for chunk in tool_calls:
                        if tool_call and chunk["index"] == tool_call["index"]:
                            if "function" in tool_call:
                                tool_call["function"]["arguments"] += cast(
                                    FunctionToolCallDelta, chunk
                                )["function"]["arguments"]
                            elif "custom" in tool_call:
                                tool_call["custom"]["input"] += cast(
                                    CustomToolCallDelta, chunk
                                )["custom"]["input"]
                        else:
                            if tool_call:
                                # this is a new tool call, yield the previous one
                                ret = consumer.send(
                                    (role, content, reasoning_content, tool_call)
                                )
                            # reset the tool call
                            tool_call = copy.deepcopy(chunk)
                elif content or reasoning_content:
                    ret = consumer.send(
                        (
                            role,
                            content,
                            reasoning_content,
                            cast(ToolCallDelta, tool_call),
                        )
                    )
            except (KeyError, IndexError):
                pass
        if tool_call:
            # yield the last tool call
            ret = consumer.send((role, None, None, tool_call))
            yield ret
        else:
            yield None
        consumer.close()
    except GeneratorExit:
        pass


def echo_consumer():
    data = None
    while True:
        data = yield data


def stream_chat_all(
    response: Iterable[ChatChunk],
) -> Generator[ChatChunkUnified, None, None]:
    producer = trans_stream_chat(
        cast(Generator[Optional[ShortChatChunk], ShortChatChunk, None], echo_consumer())
    )
    next(producer)  # prime the generator
    for data in response:
        ret = producer.send(data)
        if ret is not None:
            role, content, reasoning_content, tool_call = ret
            yield {
                "role": role,
                "content": content,
                "reasoning_content": reasoning_content,
                "tool_call": tool_call,
            }
    ret = producer.send(None)  # signal the end of the stream
    if ret is not None:
        role, content, reasoning_content, tool_call = ret
        yield {
            "role": role,
            "content": content,
            "reasoning_content": reasoning_content,
            "tool_call": tool_call,
        }
    producer.close()


def stream_chat_with_role(response: Iterable[ChatChunk]):
    for chunk in stream_chat_all(response):
        if chunk["content"]:
            yield chunk["role"], chunk["content"]


def stream_chat_with_reasoning(response: Iterable[ChatChunk]):
    for chunk in stream_chat_all(response):
        if chunk["reasoning_content"] or chunk["content"]:
            yield chunk["reasoning_content"], chunk["content"]


def stream_chat(response: Iterable[ChatChunk]):
    for _, text in stream_chat_with_role(response):
        yield text


def stream_completions(response: Iterable[CompletionsChunk]):
    for data in response:
        try:
            yield cast(str, data["choices"][0]["text"])
        except (KeyError, IndexError):
            pass


async def astream_chat_all(
    response: AsyncIterable[ChatChunk],
) -> AsyncGenerator[ChatChunkUnified, None]:
    producer = trans_stream_chat(
        cast(Generator[Optional[ShortChatChunk], ShortChatChunk, None], echo_consumer())
    )
    next(producer)  # prime the generator
    async for data in response:
        ret = producer.send(data)
        if ret is not None:
            role, content, reasoning_content, tool_call = ret
            yield {
                "role": role,
                "content": content,
                "reasoning_content": reasoning_content,
                "tool_call": tool_call,
            }
    ret = producer.send(None)  # signal the end of the stream
    if ret is not None:
        role, content, reasoning_content, tool_call = ret
        yield {
            "role": role,
            "content": content,
            "reasoning_content": reasoning_content,
            "tool_call": tool_call,
        }
    producer.close()


async def astream_chat_with_role(response: AsyncIterable[ChatChunk]):
    async for chunk in astream_chat_all(response):
        if chunk["content"]:
            yield chunk["role"], chunk["content"]


async def astream_chat_with_reasoning(response: AsyncIterable[ChatChunk]):
    async for chunk in astream_chat_all(response):
        if chunk["reasoning_content"] or chunk["content"]:
            yield chunk["reasoning_content"], chunk["content"]


async def astream_chat(response: AsyncIterable[ChatChunk]):
    async for _, text in astream_chat_with_role(response):
        yield text


async def astream_completions(response: AsyncIterable[CompletionsChunk]):
    async for data in response:
        try:
            yield cast(str, data["choices"][0]["text"])
        except (KeyError, IndexError):
            pass


def stream_to_fd(response: Iterable[bytes], fd: IO):
    for data in response:
        fd.write(data)


def stream_to_file(response: Iterable[bytes], file_path: PathType):
    with open(file_path, "wb") as f:
        stream_to_fd(response, f)


async def astream_to_fd(response: AsyncIterable[bytes], fd: IO):
    async for data in response:
        fd.write(data)


async def astream_to_file(response: AsyncIterable[bytes], file_path: PathType):
    with open(file_path, "wb") as f:
        await astream_to_fd(response, f)


def VM(**kwargs: str):
    # transform kwargs to a variable map dict
    # change each key to a % wrapped string
    transformed_vm = {f"%{key}%": value for key, value in kwargs.items()}
    return transformed_vm


def encode_bin_file(local_path: PathType):
    """
    Convert a local binary file to a base64 string.
    """
    with open(local_path, "rb") as local_file:
        return base64.b64encode(local_file.read()).decode("utf-8")


def get_mime_type(filename: str):
    """
    Get the mime type of a file based on its name.
    """
    mime_type, _ = mimetypes.guess_type(filename)
    if mime_type is None:
        return "application/octet-stream"
    return mime_type


def file_uri_to_base64(url: str, base_path: Optional[PathType]):
    """
    Convert a file URI like `file:///path/to/file` to a base64 string.
    """
    parsed = urlparse(url)
    local_path = Path(url2pathname(parsed.netloc + parsed.path))
    if base_path:
        # support relative path
        local_path = base_path / local_path
    base64_str = encode_bin_file(local_path.resolve())
    return base64_str, local_path


def file_uri_to_base64_mime(url: str, base_path: Optional[PathType]):
    """
    Convert a file URI like `file:///path/to/file` to a base64 string with mime type prefix.
    """
    base64_str, local_path = file_uri_to_base64(url, base_path)
    mime_type = get_mime_type(local_path.name)
    return f"data:{mime_type};base64,{base64_str}", local_path


def content_part_text(text: str) -> TextContentPart:
    return {"type": "text", "text": text}


def content_part_image(
    url_or_base64: str, detail: Optional[str] = None
) -> ImageContentPart:
    ret: ImageContentPart = {"type": "image_url", "image_url": {"url": url_or_base64}}
    if detail:
        ret["image_url"]["detail"] = detail
    return ret


def content_part_audio(url_or_base64: str, format: str) -> AudioContentPart:
    return {
        "type": "input_audio",
        "input_audio": {"data": url_or_base64, "format": format},
    }


def content_part_file(
    file_data: Optional[str] = None,
    file_id: Optional[str] = None,
    filename: Optional[str] = None,
) -> FileContentPart:
    file_obj: FileObject = {}
    if file_data is not None:
        file_obj["file_data"] = file_data
    if file_id is not None:
        file_obj["file_id"] = file_id
    if filename is not None:
        file_obj["filename"] = filename
    return {
        "type": "file",
        "file": file_obj,
    }
