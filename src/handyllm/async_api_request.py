import asyncio
import httpx
import json
import logging

from . import _API_TYPES_AZURE


module_logger = logging.getLogger(__name__)
module_logger.addHandler(logging.NullHandler())


async def api_request(
    url, 
    api_key, 
    organization=None, 
    api_type=None, 
    method='post', 
    timeout=None, 
    files=None, 
    raw_response=False, 
    dest_url=None, 
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
    if stream and raw_response:
        raise Exception("Cannot use 'raw_response' in stream mode.")

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
    if dest_url is not None:
        headers['Destination-URL'] = dest_url
    if method == 'post':
        if files is None:
            headers['Content-Type'] = 'application/json'
            json_data = kwargs
        else:  ## if files is not None, let httpx handle the content type
            data = kwargs
    if method == 'get' and stream:
        params['stream'] = 'true'

    # async with httpx.AsyncClient() as client:
    client = httpx.AsyncClient()
    request = client.build_request(
        method,
        url,
        headers=headers,
        data=data,
        json=json_data,
        files=files,
        params=params,
        timeout=timeout,
    )
    response = await client.send(
        request=request,
        stream=stream,
    )
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as e:
        module_logger.debug("Encountered httpx.HTTPStatusError", exc_info=True)
        raise e
        
    # if not 200 <= response.status_code < 300:
    #     # report both status code and error message
    #     try:
    #         message = response.json()['error']['message']
    #     except:
    #         message = response.text
    #     err_msg = f"OpenAI API error ({url} {response.status_code} {response.reason}): {message}"
    #     module_logger.error(err_msg)
    #     raise Exception(err_msg)

    if stream:
        return _gen_stream_response(client, response)
    else:
        await client.aclose()
        if raw_response:
            return response
        else:
            return response.json()

async def _gen_stream_response(client: httpx.AsyncClient, response: httpx.Response):
    try:
        async for raw_line in response.aiter_lines():  # do not auto decode
            if raw_line:
                if raw_line.strip() == "data: [DONE]":
                    return
                if raw_line.startswith("data: "):
                    line = raw_line[len("data: "):]
                    yield json.loads(line)
    finally:
        await response.aclose()
        await client.aclose()

async def poll(
    url, 
    method, 
    until, 
    failed, 
    interval, 
    headers=None,
    params=None,
    ):
    async with httpx.AsyncClient() as client:
        response = await client.request(method, url, headers=headers, params=params)
        if failed(response):
            return response
        while not until(response):
            await asyncio.sleep(interval(response))
            response = await client.request(method, url, headers=headers, params=params)
            if failed(response):
                return response
        return response
