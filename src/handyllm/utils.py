import requests
from urllib.parse import urlparse
import os
import time
import collections.abc
import json
import copy


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

def isiterable(arg):
    return (
        isinstance(arg, collections.abc.Iterable) 
        and not isinstance(arg, str)
    )

def join_url(base_url, *args):
    url = base_url.rstrip('/')
    for arg in args:
        url += '/' + arg.lstrip('/')
    return url

def wrap_log_input(input_content: str, log_marks, kwargs):
    arguments = copy.deepcopy(kwargs)
    # check if log_marks is iterable
    if isiterable(log_marks):
        input_lines = [str(item) for item in log_marks]
    else:
        input_lines = [str(log_marks)]
    input_lines.append(json.dumps(arguments, indent=2, ensure_ascii=False))
    input_lines.append(" INPUT START ".center(50, '-'))
    input_lines.append(input_content)
    input_lines.append(" INPUT END ".center(50, '-')+"\n")
    input_str = "\n".join(input_lines)
    return input_str

def log_result(logger, tag: str, duration: float, log_marks, kwargs, input_content: str, output_content: str):
    input_str = wrap_log_input(input_content, log_marks, kwargs)
    ## log this on result
    log_strs = []
    log_strs.append(f"{tag} result ({duration:.2f}s)")
    log_strs.append(input_str)
    log_strs.append(" OUTPUT START ".center(50, '-'))
    log_strs.append(output_content)
    log_strs.append(" OUTPUT END ".center(50, '-')+"\n")
    logger.info('\n'.join(log_strs))

def log_exception(logger, tag: str, duration: float, log_marks, kwargs, input_content: str, err_msg: str):
    input_str = wrap_log_input(input_content, log_marks, kwargs)
    ## log this on exception
    log_strs = []
    log_strs.append(f"{tag} error ({duration:.2f}s)")
    log_strs.append(input_str)
    log_strs.append(" EXCEPTION START ".center(50, '-'))
    log_strs.append(err_msg)
    log_strs.append(" EXCEPTION END ".center(50, '-')+"\n")
    logger.error('\n'.join(log_strs))

def exception2err_msg(exception: Exception):
    err_msg = f"Exception: {type(exception).__module__}.{type(exception).__name__}"
    err_msg += f"\nDetailed info: {repr(exception)}"
    if exception.args:
        print(f"\nException arguments: {exception.args}")
    return err_msg
