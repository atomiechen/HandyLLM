from __future__ import annotations
__all__ = [
    'Requestor',
    'DictRequestor',
    'BinRequestor',
    'ChatRequestor',
    'CompletionsRequestor',
]

from typing import AsyncGenerator, Callable, Generator, Generic, Optional, TypeVar, Union, cast
import asyncio
import logging
import json
import time
import requests
import httpx

from ._constants import API_TYPES_AZURE
from .response import ChatChunk, CompletionsChunk, DictProxy, ChatResponse, CompletionsResponse


module_logger = logging.getLogger(__name__)
module_logger.addHandler(logging.NullHandler())

ResponseType = TypeVar('ResponseType')
YieldType = TypeVar('YieldType')
DictResponseType = TypeVar('DictResponseType', bound='DictProxy')
DictYieldType = TypeVar('DictYieldType', bound='DictProxy')


class Requestor(Generic[ResponseType, YieldType]):
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
        azure_poll=False, 
        raw=False, 
        chunk_size=1024, 
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
        self.azure_poll = azure_poll
        self.raw = raw
        self.chunk_size = chunk_size
        self.dest_url = dest_url

        self._stream = cast(bool, kwargs.get('stream', False))
        if self._stream and azure_poll:
            raise Exception("Cannot use 'azure_poll' in stream mode.")

        self.headers = {}
        self.json_data = None
        self.data = None
        self.params = {}
        if api_type in API_TYPES_AZURE:
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
        self._change_stream_mode(self._stream)

    def _change_stream_mode(self, stream: bool):
        self._stream = stream
        if stream:
            if self.method == 'post':
                if self.files is None:
                    dict_data = self.json_data
                else:
                    dict_data = self.data
                assert isinstance(dict_data, dict)
                dict_data['stream'] = True
            elif self.method == 'get':
                self.params['stream'] = 'true'
        else:
            if self.method == 'post':
                if self.files is None:
                    dict_data = self.json_data
                else:
                    dict_data = self.data
                assert isinstance(dict_data, dict)
                dict_data.pop('stream', None)
            elif self.method == 'get':
                self.params.pop('stream', None)
    
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

    def _check_image_error(self, response):
        response_dict = response.json()
        if response_dict['status'] == 'failed':
            err_msg = f"Image generation failed: {response_dict['error']['code']} {response_dict['error']['message']}"
            module_logger.error(err_msg)
            raise Exception(err_msg)
    
    def _check_image_end(self, response):
        return response.json()['status'] in ['succeeded', 'failed']
    
    def _get_image_retry(self, response):
        try:
            return int(response.headers.get('retry-after'))
        except:
            return 1
    
    def _check_timeout(self, timeout_ddl):
        if timeout_ddl and time.perf_counter() > timeout_ddl:
            raise Exception("Timeout")
    
    def _make_wrapped_exception(self, response: Union[requests.Response, httpx.Response]):
        # report both status code and error message
        try:
            # message = response.json()['error']['message']
            message = response.json()
        except:
            message = response.text
        reason = response.reason if isinstance(response, requests.Response) else response.reason_phrase
        err_msg = f"API error ({self.url} {response.status_code} {reason}) - {message}"
        return Exception(err_msg)

    def stream(self) -> Generator[YieldType, None, None]:
        '''
        Request in stream mode, will return a generator.
        '''
        self._change_stream_mode(True)
        return cast(Generator, self.call())

    def fetch(self) -> ResponseType:
        '''
        Request in non-stream mode, will not return until the response is complete.
        '''
        self._change_stream_mode(False)
        return cast(ResponseType, self.call())

    def call(self) -> Union[ResponseType, Generator[YieldType, None, None]]:
        '''
        Execute the request. Stream or non-stream mode depends on the stream parameter.
        '''
        if self._sync_client is None:
            raise Exception("Sync request client is not set")

        if self._prepare_callback:
            prepare_ret = self._prepare_callback()
        else:
            prepare_ret = None
        timeout_ddl = time.perf_counter()+self.timeout if self.timeout else None
        self._log_request()
        try:
            raw_response = self._call_raw()

            if self._stream:
                if self.raw:
                    response = self._gen_stream_bin_response(raw_response, prepare_ret)
                else:
                    response = self._gen_stream_response(raw_response, prepare_ret)
            else:
                if self.azure_poll:
                    poll_url = raw_response.headers['operation-location']
                    response = self.poll(poll_url, timeout_ddl=timeout_ddl).json()
                    response = response.get('result', response)
                elif self.raw:
                    response = raw_response.content
                else:
                    response = raw_response.json()
            
            if self._response_callback:
                response = self._response_callback(response, prepare_ret)
            return cast(Union[ResponseType, Generator[YieldType, None, None]], response)
        except Exception as e:
            if self._exception_callback:
                self._exception_callback(e, prepare_ret)
            raise e

    def _call_raw(self) -> requests.Response:
        self._sync_client = cast(requests.Session, self._sync_client)
        response = self._sync_client.request(
            self.method,
            self.url,
            headers=self.headers,
            data=self.data,
            json=self.json_data,
            files=self.files,
            params=self.params,
            stream=self._stream,
            timeout=self.timeout,
            )
        try:
            response.raise_for_status()
        except requests.HTTPError as e:
            module_logger.error(e, exc_info=True)
            # raise a new exception
            raise self._make_wrapped_exception(response) from None
        return response

    def _gen_stream_response(self, raw_response: requests.Response, prepare_ret):
        with raw_response:
            try:
                for byte_line in raw_response.iter_lines():  # do not auto decode
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
    
    def _gen_stream_bin_response(self, raw_response: requests.Response, prepare_ret):
        with raw_response:
            try:
                for chunk in raw_response.iter_content(chunk_size=self.chunk_size):
                    yield chunk
            except Exception as e:
                if self._exception_callback:
                    self._exception_callback(e, prepare_ret)
                raise e

    def poll(self, url, timeout_ddl=None, params=None) -> requests.Response:
        self._sync_client = cast(requests.Session, self._sync_client)
        self._check_timeout(timeout_ddl)
        headers= { "api-key": self.api_key, "Content-Type": "application/json" }
        response = self._sync_client.request('get', url, headers=headers, params=params)
        self._check_image_error(response)
        while not self._check_image_end(response):
            self._check_timeout(timeout_ddl)
            time.sleep(self._get_image_retry(response))
            response = self._sync_client.request('get', url, headers=headers, params=params)
        self._check_image_error(response)
        return response

    async def astream(self) -> AsyncGenerator[YieldType, None]:
        '''
        Request in stream mode asynchronously, will return an async generator.
        '''
        self._change_stream_mode(True)
        return cast(AsyncGenerator, await self.acall())

    async def afetch(self) -> ResponseType:
        '''
        Request in non-stream mode asynchronously, will not return until the response is complete.
        '''
        self._change_stream_mode(False)
        return cast(ResponseType, await self.acall())

    async def acall(self) -> Union[ResponseType, AsyncGenerator[YieldType, None]]:
        '''
        Execute the request asynchronously. Stream or non-stream mode depends on the stream parameter.
        '''
        if self._async_client is None:
            raise Exception("Async request client is not set")

        if self._prepare_callback:
            prepare_ret = self._prepare_callback()
        else:
            prepare_ret = None
        timeout_ddl = time.perf_counter()+self.timeout if self.timeout else None
        self._log_request()
        try:
            raw_response = await self._acall_raw()

            if self._stream:
                if self.raw:
                    response = self._agen_stream_bin_response(raw_response, prepare_ret)
                else:
                    response = self._agen_stream_response(raw_response, prepare_ret)
            else:
                if self.azure_poll:
                    poll_url = raw_response.headers['operation-location']
                    response = (await self.apoll(poll_url, timeout_ddl=timeout_ddl)).json()
                    response = response.get('result', response)
                elif self.raw:
                    response = raw_response.content
                else:
                    response = raw_response.json()
            
            if self._response_callback:
                response = self._response_callback(response, prepare_ret)
            return cast(Union[ResponseType, AsyncGenerator[YieldType, None]], response)
        except Exception as e:
            if self._exception_callback:
                self._exception_callback(e, prepare_ret)
            raise e

    async def _acall_raw(self):
        self._async_client = cast(httpx.AsyncClient, self._async_client)
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
            stream=self._stream,
        )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            module_logger.error(e, exc_info=True)
            # read the response body first to prevent httpx.ResponseNotRead exception
            # ref: https://github.com/encode/httpx/discussions/1856#discussioncomment-1316674
            await response.aread()
            # raise a new exception
            raise self._make_wrapped_exception(response) from None
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

    async def _agen_stream_bin_response(self, raw_response: httpx.Response, prepare_ret):
        try:
            async for chunk in raw_response.aiter_bytes():
                yield chunk
        except Exception as e:
            if self._exception_callback:
                self._exception_callback(e, prepare_ret)
            raise e
        finally:
            await raw_response.aclose()

    async def apoll(self, url, timeout_ddl=None, params=None) -> httpx.Response:
        self._async_client = cast(httpx.AsyncClient, self._async_client)
        self._check_timeout(timeout_ddl)
        headers= { "api-key": self.api_key, "Content-Type": "application/json" }
        response = await self._async_client.request('get', url, headers=headers, params=params)
        self._check_image_error(response)
        while not self._check_image_end(response):
            self._check_timeout(timeout_ddl)
            await asyncio.sleep(self._get_image_retry(response))
            response = await self._async_client.request('get', url, headers=headers, params=params)
        self._check_image_error(response)
        return response
    
    def set_sync_client(self, client: Optional[requests.Session]):
        self._sync_client = client
    
    def set_async_client(self, client: Optional[httpx.AsyncClient]):
        self._async_client = client
    
    def set_prepare_callback(self, func: Callable):
        self._prepare_callback = func
    
    def set_response_callback(self, func: Callable):
        self._response_callback = func

    def set_exception_callback(self, func: Callable):
        self._exception_callback = func


class DictRequestor(Requestor[DictResponseType, DictYieldType], Generic[DictResponseType, DictYieldType]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.raw = False
    
    def call(self) -> Union[DictResponseType, Generator[DictYieldType, None, None]]:
        response = super().call()
        if self._stream:
            def gen_wrapper(response):
                for data in response:
                    yield DictProxy(data)
            return cast(Generator[DictYieldType, None, None], gen_wrapper(response))
        else:
            return cast(DictResponseType, DictProxy(response))
    
    async def acall(self) -> Union[DictResponseType, AsyncGenerator[DictYieldType, None]]:
        response = await super().acall()
        if self._stream:
            async def agen_wrapper(response):
                async for data in response:
                    yield DictProxy(data)
            return cast(AsyncGenerator[DictYieldType, None], agen_wrapper(response))
        else:
            return cast(DictResponseType, DictProxy(response))


class BinRequestor(Requestor[bytes, bytes]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.raw = True


class ChatRequestor(DictRequestor[ChatResponse, ChatChunk]):
    pass


class CompletionsRequestor(DictRequestor[CompletionsResponse, CompletionsChunk]):
    pass

