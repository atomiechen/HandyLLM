import base64
from typing import IO, AsyncIterable, Iterable, Optional, cast
from urllib.parse import urlparse
import os
import time

from .types import PathType


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

def stream_chat_all(response: Iterable[dict]):
    role = ''
    tool_call = {}
    for data in response:
        try:
            message = data['choices'][0]['delta']
            if 'role' in message:
                role = cast(str, message['role'])
            content = cast(Optional[str], message.get('content'))
            tool_calls = cast(Optional[list], message.get('tool_calls'))
            if tool_calls:
                for chunk in tool_calls:
                    chunk = cast(dict, chunk)
                    if chunk['index'] == tool_call.get('index'):
                        tool_call['function']['arguments'] += chunk['function']['arguments']
                    else:
                        # this is a new tool call, yield the previous one
                        yield role, content, tool_call
                        # reset the tool call
                        tool_call = chunk
            elif content:
                yield role, content, tool_call
        except (KeyError, IndexError):
            pass
    if tool_call:
        # yield the last tool call
        yield role, None, tool_call

def stream_chat_with_role(response: Iterable[dict]):
    for role, text, _ in stream_chat_all(response):
        if text:
            yield role, text

def stream_chat(response: Iterable[dict]):
    for _, text in stream_chat_with_role(response):
        yield text

def stream_completions(response: Iterable[dict]):
    for data in response:
        try:
            yield data['choices'][0]['text']
        except (KeyError, IndexError):
            pass

async def astream_chat_all(response: AsyncIterable[dict]):
    role = ''
    tool_call = {}
    async for data in response:
        try:
            message = data['choices'][0]['delta']
            if 'role' in message:
                role = cast(str, message['role'])
            content = cast(Optional[str], message.get('content'))
            tool_calls = message.get('tool_calls')
            if tool_calls:
                for chunk in tool_calls:
                    chunk = cast(dict, chunk)
                    if chunk['index'] == tool_call.get('index'):
                        tool_call['function']['arguments'] += chunk['function']['arguments']
                    else:
                        # this is a new tool call, yield the previous one
                        yield role, content, tool_call
                        # reset the tool call
                        tool_call = chunk
            elif content:
                yield role, content, tool_call
        except (KeyError, IndexError):
            pass
    if tool_call:
        # yield the last tool call
        yield role, None, tool_call

async def astream_chat_with_role(response: AsyncIterable[dict]):
    async for role, text, _ in astream_chat_all(response):
        if text:
            yield role, text

async def astream_chat(response: AsyncIterable[dict]):
    async for _, text in astream_chat_with_role(response):
        yield text

async def astream_completions(response: AsyncIterable[dict]):
    async for data in response:
        try:
            yield data['choices'][0]['text']
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

def VM(**kwargs):
    # transform kwargs to a variable map dict
    # change each key to a % wrapped string
    transformed_vm = {f'%{key}%': value for key, value in kwargs.items()}
    return transformed_vm

def encode_image(image_path: PathType):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

