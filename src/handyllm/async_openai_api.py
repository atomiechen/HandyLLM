import time

from .base_openai_api import BaseOpenAIAPI
from .async_api_request import api_request, poll
from . import utils

from . import _API_TYPES_AZURE


class AsyncOpenAIAPI(BaseOpenAIAPI):

    @classmethod
    async def api_request_endpoint(
        cls,
        request_url, 
        **kwargs
        ):
        api_key, organization, api_base, api_type, api_version, engine, dest_url = cls.consume_kwargs(kwargs)
        url = utils.join_url(api_base, request_url)
        return await api_request(url, api_key, organization=organization, api_type=api_type, dest_url=dest_url, **kwargs)

    @staticmethod
    async def stream_chat_with_role(response):
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
    
    @classmethod
    async def stream_chat(cls, response):
        async for _, text in cls.stream_chat_with_role(response):
            yield text
    
    @staticmethod
    async def stream_completions(response):
        async for data in response:
            try:
                yield data['choices'][0]['text']
            except (KeyError, IndexError):
                pass
    
    @classmethod
    async def chat(cls, messages, logger=None, log_marks=[], **kwargs):
        api_key, organization, api_base, api_type, api_version, engine, dest_url = cls.consume_kwargs(kwargs)
        start_time = time.perf_counter()
        try:
            response = await cls.api_request_endpoint(
                cls.get_chat_request_url(api_type, api_version, engine), 
                messages=messages, 
                method='post', 
                api_key=api_key,
                organization=organization,
                api_base=api_base,
                api_type=api_type,
                dest_url=dest_url,
                **kwargs
            )
            stream = kwargs.get('stream', False)
            response = cls._chat_log_response(logger, log_marks, kwargs, messages, start_time, response, stream)
            return response
        except Exception as e:
            cls._chat_log_exception(logger, log_marks, kwargs, messages, start_time, e)
            raise e


    @classmethod
    async def completions(cls, prompt, logger=None, log_marks=[], **kwargs):
        api_key, organization, api_base, api_type, api_version, engine, dest_url = cls.consume_kwargs(kwargs)
        input_str = cls._completions_log_input(prompt, logger, log_marks, kwargs)
        start_time = time.time()
        try:
            response = await cls.api_request_endpoint(
                cls.get_completions_request_url(api_type, api_version, engine), 
                prompt=prompt, 
                method='post', 
                api_key=api_key,
                organization=organization,
                api_base=api_base,
                api_type=api_type,
                dest_url=dest_url,
                **kwargs
            )
            stream = kwargs.get('stream', False)
            response = cls._completions_log_output(response, input_str, start_time, logger, stream)
            return response
        except Exception as e:
            cls._completions_log_exception(e, input_str, start_time, logger)
            raise e

