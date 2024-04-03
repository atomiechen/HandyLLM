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

def stream_chat_with_role(response):
    role = ''
    for data in response:
        try:
            message = data['choices'][0]['delta']
            if 'role' in message:
                role = message['role']
            if 'content' in message:
                text = message['content']
                yield role, text
        except (KeyError, IndexError):
            pass

def stream_chat(response):
    for _, text in stream_chat_with_role(response):
        yield text

def stream_completions(response):
    for data in response:
        try:
            yield data['choices'][0]['text']
        except (KeyError, IndexError):
            pass

async def astream_chat_with_role(response):
    role = ''
    async for data in response:
        try:
            message = data['choices'][0]['delta']
            if 'role' in message:
                role = message['role']
            if 'content' in message:
                text = message['content']
                yield role, text
        except (KeyError, IndexError):
            pass

async def astream_chat(response):
    async for _, text in astream_chat_with_role(response):
        yield text

async def astream_completions(response):
    async for data in response:
        try:
            yield data['choices'][0]['text']
        except (KeyError, IndexError):
            pass

