import requests
import json
import logging
import time

from . import _API_TYPES_AZURE


module_logger = logging.getLogger(__name__)


def api_request(
    url, 
    api_key, 
    organization=None, 
    api_type=None, 
    method='post', 
    timeout=None, 
    files=None, 
    raw_response=False, 
    **kwargs):
    if api_key is None:
        raise Exception("OpenAI API key is not set")
    if url is None:
        raise Exception("OpenAI API url is not set")
    if api_type is None:
        raise Exception("OpenAI API type is not set")

    ## log request info
    log_strs = []
    # avoid logging the whole api_key
    plaintext_len = 8
    log_strs.append(f"API request {url}")
    log_strs.append(f"api_type: {api_type}")
    log_strs.append(f"api_key: {api_key[:plaintext_len]}{'*'*(len(api_key)-plaintext_len)}")
    if organization is not None:
        log_strs.append(f"organization: {organization[:plaintext_len]}{'*'*(len(organization)-plaintext_len)}")
    log_strs.append(f"timeout: {timeout}")
    module_logger.info('\n'.join(log_strs))

    stream = kwargs.get('stream', False)
    headers = {}
    json_data = None
    data = None
    params = {}
    if api_type in _API_TYPES_AZURE:
        headers['api-key'] = api_key
    else:
        headers['Authorization'] = 'Bearer ' + api_key
    if organization is not None:
        headers['OpenAI-Organization'] = organization
    if method == 'post':
        if files is None:
            headers['Content-Type'] = 'application/json'
            json_data = kwargs
        else:  ## if files is not None, let requests handle the content type
            data = kwargs
    if method == 'get' and stream:
        params['stream'] = 'true'

    response = requests.request(
        method,
        url,
        headers=headers,
        data=data,
        json=json_data,
        files=files,
        params=params,
        stream=stream,
        timeout=timeout,
        )
    if not 200 <= response.status_code < 300:
        # report both status code and error message
        try:
            message = response.json()['error']['message']
        except:
            message = response.text
        err_msg = f"OpenAI API error ({url} {response.status_code} {response.reason}): {message}"
        module_logger.error(err_msg)
        raise Exception(err_msg)

    if raw_response:
        return response
    elif stream:
        return _gen_stream_response(response)
    else:
        return response.json()

def _gen_stream_response(response):
    for byte_line in response.iter_lines():  # do not auto decode
        if byte_line:
            if byte_line.strip() == b"data: [DONE]":
                return
            if byte_line.startswith(b"data: "):
                line = byte_line[len(b"data: "):].decode("utf-8")
                yield json.loads(line)

def poll(
    url, 
    method, 
    until, 
    failed, 
    interval, 
    headers=None,
    params=None,
    ):
    response = requests.request(method, url, headers=headers, params=params)
    if failed(response):
        return response
    while not until(response):
        time.sleep(interval(response))
        response = requests.request(method, url, headers=headers, params=params)
        if failed(response):
            return response
    return response
