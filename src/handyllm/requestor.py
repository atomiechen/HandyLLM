import logging
import json
import requests
import httpx

from . import _API_TYPES_AZURE


module_logger = logging.getLogger(__name__)
module_logger.addHandler(logging.NullHandler())


class Requestor:
    def __init__(
        self, 
        api_type, 
        url, 
        api_key, 
        *,
        organization=None, 
        method='post', 
        timeout=None, 
        files=None, 
        raw_response=False, 
        dest_url=None, 
        **kwargs) -> None:
        
        self._sync_client = None
        self._async_client = None
        self._prepare_callback = None
        self._response_callback = None
        self._exception_callback = None
        
        if api_type is None:
            raise Exception("OpenAI API type is not set")
        else:
            self.api_type = api_type
        if url is None:
            raise Exception("OpenAI API url is not set")
        else:
            self.url = url
        if api_key is None:
            raise Exception("OpenAI API key is not set")
        else:
            self.api_key = api_key
        self.organization = organization
        self.method = method
        self.timeout = timeout
        self.files = files
        self.raw_response = raw_response
        self.dest_url = dest_url

        self.stream = kwargs.get('stream', False)
        if self.stream and raw_response:
            raise Exception("Cannot use 'raw_response' in stream mode.")

        self.headers = {}
        self.json_data = None
        self.data = None
        self.params = {}
        if api_type in _API_TYPES_AZURE:
            self.headers['api-key'] = api_key
        else:
            self.headers['Authorization'] = 'Bearer ' + api_key
        if organization is not None:
            self.headers['OpenAI-Organization'] = organization
        if dest_url is not None:
            self.headers['Destination-URL'] = dest_url
        if method == 'post':
            if files is None:
                self.headers['Content-Type'] = 'application/json'
                self.json_data = kwargs
            else:  ## if files is not None, let httpx handle the content type
                self.data = kwargs
        if method == 'get' and self.stream:
            self.params['stream'] = 'true'
    
    def _log_request(self):
        ## log request info
        log_strs = []
        # avoid logging the whole api_key
        plaintext_len = 8
        log_strs.append(f"API request {self.url}")
        log_strs.append(f"api_type: {self.api_type}")
        log_strs.append(f"api_key: {self.api_key[:plaintext_len]}{'*'*(len(self.api_key)-plaintext_len)}")
        if self.organization is not None:
            log_strs.append(f"organization: {self.organization[:plaintext_len]}{'*'*(len(self.organization)-plaintext_len)}")
        log_strs.append(f"timeout: {self.timeout}")
        module_logger.info('\n'.join(log_strs))

    def call(self):
        if self._sync_client is None:
            raise Exception("Sync request client is not set")

        if self._prepare_callback:
            prepare_ret = self._prepare_callback()
        else:
            prepare_ret = None

        self._log_request()
        try:
            raw_response = self._call_raw()

            if self.stream:
                response = self._gen_stream_response(raw_response, prepare_ret)
            else:
                if self.raw_response:
                    response = raw_response
                else:
                    response = raw_response.json()
            
            if self._response_callback:
                response = self._response_callback(response, prepare_ret)
            return response
        except Exception as e:
            if self._exception_callback:
                self._exception_callback(e, prepare_ret)
            raise e

    def _call_raw(self):
        response = self._sync_client.request(
            self.method,
            self.url,
            headers=self.headers,
            data=self.data,
            json=self.json_data,
            files=self.files,
            params=self.params,
            stream=self.stream,
            timeout=self.timeout,
            )
        if not 200 <= response.status_code < 300:
            # report both status code and error message
            try:
                message = response.json()['error']['message']
            except:
                message = response.text
            err_msg = f"OpenAI API error ({self.url} {response.status_code} {response.reason}): {message}"
            module_logger.error(err_msg)
            raise Exception(err_msg)
        return response

    def _gen_stream_response(self, response: requests.Response, prepare_ret):
        with response:
            try:
                for byte_line in response.iter_lines():  # do not auto decode
                    if byte_line:
                        if byte_line.strip() == b"data: [DONE]":
                            return
                        if byte_line.startswith(b"data: "):
                            line = byte_line[len(b"data: "):].decode("utf-8")
                            yield json.loads(line)
            except Exception as e:
                if self._exception_callback:
                    self._exception_callback(e, prepare_ret)
                raise e

    async def acall(self):
        if self._async_client is None:
            raise Exception("Async request client is not set")

        if self._prepare_callback:
            prepare_ret = self._prepare_callback()
        else:
            prepare_ret = None

        self._log_request()
        try:
            raw_response = await self._acall_raw()

            if self.stream:
                response = self._agen_stream_response(raw_response, prepare_ret)
            else:
                if self.raw_response:
                    response = raw_response
                else:
                    response = raw_response.json()
            
            if self._response_callback:
                response = self._response_callback(response, prepare_ret)
            return response
        except Exception as e:
            if self._exception_callback:
                self._exception_callback(e, prepare_ret)
            raise e

    async def _acall_raw(self):
        request = self._async_client.build_request(
            self.method,
            self.url,
            headers=self.headers,
            data=self.data,
            json=self.json_data,
            files=self.files,
            params=self.params,
            timeout=self.timeout,
        )
        response = await self._async_client.send(
            request=request,
            stream=self.stream,
        )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            module_logger.error("Encountered httpx.HTTPStatusError", exc_info=True)
            raise e
        return response

    async def _agen_stream_response(self, raw_response: httpx.Response, prepare_ret):
        try:
            async for raw_line in raw_response.aiter_lines():
                if raw_line:
                    if raw_line.strip() == "data: [DONE]":
                        return
                    if raw_line.startswith("data: "):
                        line = raw_line[len("data: "):]
                        yield json.loads(line)
        except Exception as e:
            if self._exception_callback:
                self._exception_callback(e, prepare_ret)
            raise e
        finally:
            await raw_response.aclose()
    
    def set_sync_client(self, client: requests.Session):
        self._sync_client = client
    
    def set_async_client(self, client: httpx.AsyncClient):
        self._async_client = client
    
    def set_prepare_callback(self, func: callable):
        self._prepare_callback = func
    
    def set_response_callback(self, func: callable):
        self._response_callback = func

    def set_exception_callback(self, func: callable):
        self._exception_callback = func
