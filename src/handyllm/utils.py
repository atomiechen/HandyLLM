import io
from urllib.parse import urlparse
import os
import time


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

def stream_chat_all(response):
    role = ''
    tool_call = {}
    for data in response:
        try:
            message = data['choices'][0]['delta']
            if 'role' in message:
                role = message['role']
            content = message.get('content')
            tool_calls = message.get('tool_calls')
            if tool_calls:
                for chunk in tool_calls:
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

def stream_chat_with_role(response):
    for role, text, _ in stream_chat_all(response):
        yield role, text

def stream_chat(response):
    for _, text in stream_chat_with_role(response):
        yield text

def stream_completions(response):
    for data in response:
        try:
            yield data['choices'][0]['text']
        except (KeyError, IndexError):
            pass

async def astream_chat_all(response):
    role = ''
    tool_call = {}
    async for data in response:
        try:
            message = data['choices'][0]['delta']
            if 'role' in message:
                role = message['role']
            content = message.get('content')
            tool_calls = message.get('tool_calls')
            if tool_calls:
                for chunk in tool_calls:
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

async def astream_chat_with_role(response):
    async for role, text, _ in astream_chat_all(response):
        yield role, text

async def astream_chat(response):
    async for _, text in astream_chat_with_role(response):
        yield text

async def astream_completions(response):
    async for data in response:
        try:
            yield data['choices'][0]['text']
        except (KeyError, IndexError):
            pass

def stream_to_fd(response, fd: io.IOBase):
    for data in response:
        fd.write(data)

def stream_to_file(response, file_path):
    with open(file_path, 'wb') as f:
        stream_to_fd(response, f)

async def astream_to_fd(response, fd: io.IOBase):
    async for data in response:
        fd.write(data)

async def astream_to_file(response, file_path):
    with open(file_path, 'wb') as f:
        await astream_to_fd(response, f)

def VM(**kwargs):
    # transform kwargs to a variable map dict
    # change each key to a % wrapped string
    transformed_vm = {f'%{key}%': value for key, value in kwargs.items()}
    return transformed_vm
