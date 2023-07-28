import requests
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

