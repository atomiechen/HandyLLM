__all__ = [
    'get_filename_from_url',
    'download_binary',
    'trans_stream_chat',
    'echo_consumer',
    'stream_chat_all',
    'stream_chat_with_role',
    'stream_chat',
    'stream_completions',
    'astream_chat_all',
    'astream_chat_with_role',
    'astream_chat',
    'astream_completions',
    'stream_to_fd',
    'stream_to_file',
    'astream_to_fd',
    'astream_to_file',
    'VM',
    'encode_image',
    'local_path_to_base64',
]

import base64
from pathlib import Path
from typing import IO, AsyncGenerator, AsyncIterable, Generator, Iterable, Optional, TypeVar, cast
from urllib.parse import urlparse
from urllib.request import url2pathname
import os
import time

from .types import PathType, ShortChatChunk
from .response import ChatChunk, CompletionsChunk, ToolCallDelta

YieldType = TypeVar('YieldType')


def get_filename_from_url(download_url):
    # Parse the URL.
    parsed_url = urlparse(download_url)
    # The last part of the path is usually the filename.
    filename = os.path.basename(parsed_url.path)
    return filename

def download_binary(download_url, file_path=None, dir='.'):
    import requests
    response = requests.get(download_url, allow_redirects=True)
    if file_path == None:
        filename = get_filename_from_url(download_url)
        if filename == '' or filename == None:
            filename = 'download_' + time.strftime("%Y%m%d_%H%M%S")
        file_path = os.path.abspath(os.path.join(dir, filename))
    # Open the file in binary mode and write to it.
    with open(file_path, "wb") as file:
        file.write(response.content)
    return file_path

def trans_stream_chat(consumer: Generator[YieldType, ShortChatChunk, None]) -> Generator[Optional[YieldType], ChatChunk, None]:
    next(consumer) # prime the generator
    role = ''
    tool_call = ToolCallDelta()
    ret = None
    try:
        while True:
            data = yield ret
            ret = None
            try:
                message = data['choices'][0]['delta']
                if 'role' in message:
                    role = cast(str, message['role'])
                content = cast(Optional[str], message.get('content'))
                tool_calls = cast(Optional[list[ToolCallDelta]], message.get('tool_calls'))
                if tool_calls:
                    for chunk in tool_calls:
                        if chunk['index'] == tool_call.get('index'):
                            tool_call['function']['arguments'] += chunk['function']['arguments']
                        else:
                            # this is a new tool call, yield the previous one
                            ret = consumer.send((role, content, tool_call))
                            # reset the tool call
                            tool_call = ToolCallDelta(chunk)
                elif content:
                    ret = consumer.send((role, content, tool_call))
            except (KeyError, IndexError):
                pass
    except GeneratorExit:
        if tool_call:
            # yield the last tool call
            ret = consumer.send((role, None, tool_call))
        consumer.close()

def echo_consumer():
    data = None
    while True:
        data = yield data

def stream_chat_all(response: Iterable[ChatChunk]) -> Generator[ShortChatChunk, None, None]:
    producer = trans_stream_chat(echo_consumer())
    next(producer) # prime the generator
    for data in response:
        ret = producer.send(data)
        if ret is not None:
            yield ret
    producer.close()

def stream_chat_with_role(response: Iterable[ChatChunk]):
    for role, text, _ in stream_chat_all(response):
        if text:
            yield role, text

def stream_chat(response: Iterable[ChatChunk]):
    for _, text in stream_chat_with_role(response):
        yield text

def stream_completions(response: Iterable[CompletionsChunk]):
    for data in response:
        try:
            yield cast(str, data['choices'][0]['text'])
        except (KeyError, IndexError):
            pass

async def astream_chat_all(response: AsyncIterable[ChatChunk]) -> AsyncGenerator[ShortChatChunk, None]:
    producer = trans_stream_chat(echo_consumer())
    next(producer) # prime the generator
    async for data in response:
        ret = producer.send(data)
        if ret is not None:
            yield ret
    producer.close()

async def astream_chat_with_role(response: AsyncIterable[ChatChunk]):
    async for role, text, _ in astream_chat_all(response):
        if text:
            yield role, text

async def astream_chat(response: AsyncIterable[ChatChunk]):
    async for _, text in astream_chat_with_role(response):
        yield text

async def astream_completions(response: AsyncIterable[CompletionsChunk]):
    async for data in response:
        try:
            yield cast(str, data['choices'][0]['text'])
        except (KeyError, IndexError):
            pass

def stream_to_fd(response: Iterable[bytes], fd: IO):
    for data in response:
        fd.write(data)

def stream_to_file(response: Iterable[bytes], file_path: PathType):
    with open(file_path, 'wb') as f:
        stream_to_fd(response, f)

async def astream_to_fd(response: AsyncIterable[bytes], fd: IO):
    async for data in response:
        fd.write(data)

async def astream_to_file(response: AsyncIterable[bytes], file_path: PathType):
    with open(file_path, 'wb') as f:
        await astream_to_fd(response, f)

def VM(**kwargs: str):
    # transform kwargs to a variable map dict
    # change each key to a % wrapped string
    transformed_vm = {f'%{key}%': value for key, value in kwargs.items()}
    return transformed_vm

def encode_image(image_path: PathType):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def local_path_to_base64(url: str, base_path: Optional[PathType]):
    # replace the image URL with the actual image
    parsed = urlparse(url)
    local_path = Path(url2pathname(parsed.netloc + parsed.path))
    if base_path:
        # support relative path
        local_path = base_path / local_path
    base64_image = encode_image(local_path.resolve())
    return f"data:image/jpeg;base64,{base64_image}"

