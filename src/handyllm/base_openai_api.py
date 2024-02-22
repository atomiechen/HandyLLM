import os
import time
from urllib.parse import quote_plus
import json
import copy
import inspect

from .api_request import api_request, poll
from .endpoint_manager import Endpoint, EndpointManager
from .prompt_converter import PromptConverter
from ._utils import join_url, log_result, log_exception, exception2err_msg

from . import _API_BASE_OPENAI, _API_TYPE_OPENAI, _API_TYPES_AZURE, _API_VERSION_AZURE


class BaseOpenAIAPI:
    
    # deprecated
    base_url = _API_BASE_OPENAI
    
    # set this to your API base;
    # or environment variable OPENAI_API_BASE will be used.
    # can be None (roll back to default).
    api_base = None
    
    # set this to your API key; 
    # or environment variable OPENAI_API_KEY will be used.
    api_key = None
    
    # set this to your organization ID; 
    # or environment variable OPENAI_ORGANIZATION will be used;
    # can be None.
    organization = None
    
    # set this to your API type;
    # or environment variable OPENAI_API_TYPE will be used;
    # can be None (roll back to default).
    api_type = None
    
    # set this to your API version;
    # or environment variable OPENAI_API_VERSION will be used;
    # can be None.
    api_version = None
    
    # set this to your model-engine map;
    # or environment variable MODEL_ENGINE_MAP will be used;
    # can be None.
    model_engine_map = None
    
    converter = PromptConverter()
    
    @classmethod
    def get_api_key(cls, api_key=None):
        return api_key or cls.api_key or os.environ.get('OPENAI_API_KEY')
    
    @classmethod
    def get_organization(cls, organization=None):
        return organization or cls.organization or os.environ.get('OPENAI_ORGANIZATION')
    
    @classmethod
    def get_api_base(cls, api_base=None):
        return api_base or cls.api_base or os.environ.get('OPENAI_API_BASE') or _API_BASE_OPENAI
    
    @classmethod
    def get_api_type_and_version(cls, api_type=None, api_version=None):
        api_type = api_type or cls.api_type or os.environ.get('OPENAI_API_TYPE') or _API_TYPE_OPENAI
        api_version = api_version or cls.api_version or os.environ.get('OPENAI_API_VERSION')
        if not api_version and api_type and api_type.lower() in _API_TYPES_AZURE:
            api_version = _API_VERSION_AZURE
        return api_type, api_version

    @classmethod
    def get_model_engine_map(cls, model_engine_map=None):
        if model_engine_map:
            return model_engine_map
        if cls.model_engine_map:
            return cls.model_engine_map
        try:
            json_str = os.environ.get('MODEL_ENGINE_MAP')
            return json.loads(json_str)
        except:
            return None

    @classmethod
    def api_request_endpoint(
        cls,
        request_url, 
        **kwargs
        ):
        api_key, organization, api_base, api_type, api_version, engine, dest_url = cls.consume_kwargs(kwargs)
        url = join_url(api_base, request_url)
        return api_request(url, api_key, organization=organization, api_type=api_type, dest_url=dest_url, **kwargs)
    
    @classmethod
    def consume_kwargs(cls, kwargs):
        api_key = organization = api_base = api_type = api_version = engine = model_engine_map = dest_url = None

        # read API info from endpoint_manager
        endpoint_manager = kwargs.pop('endpoint_manager', None)
        if endpoint_manager is not None:
            if not isinstance(endpoint_manager, EndpointManager):
                raise Exception("endpoint_manager must be an instance of EndpointManager")
            # get_next_endpoint() will be called once for each request
            api_key, organization, api_base, api_type, api_version, model_engine_map, dest_url = endpoint_manager.get_next_endpoint().get_api_info()

        # read API info from endpoint (override API info from endpoint_manager)
        endpoint = kwargs.pop('endpoint', None)
        if endpoint is not None:
            if not isinstance(endpoint, Endpoint):
                raise Exception("endpoint must be an instance of Endpoint")
            api_key, organization, api_base, api_type, api_version, model_engine_map, dest_url = endpoint.get_api_info()

        # read API info from kwargs, class variables, and environment variables
        api_key = cls.get_api_key(kwargs.pop('api_key', api_key))
        organization = cls.get_organization(kwargs.pop('organization', organization))
        api_base = cls.get_api_base(kwargs.pop('api_base', api_base))
        api_type, api_version = cls.get_api_type_and_version(
            kwargs.pop('api_type', api_type), 
            kwargs.pop('api_version', api_version)
        )
        model_engine_map = cls.get_model_engine_map(kwargs.pop('model_engine_map', model_engine_map))

        deployment_id = kwargs.pop('deployment_id', None)
        engine = kwargs.pop('engine', deployment_id)
        # if using Azure and engine not provided, try to get it from model parameter
        if api_type and api_type.lower() in _API_TYPES_AZURE:
            model = kwargs.pop('model', None)
            if not engine and model:
                if model_engine_map:
                    engine = model_engine_map.get(model, model)
                else:
                    engine = model
        dest_url = kwargs.pop('dest_url', dest_url)
        return api_key, organization, api_base, api_type, api_version, engine, dest_url

    @staticmethod
    def get_request_url(request_url, api_type, api_version, engine):
        if api_type and api_type.lower() in _API_TYPES_AZURE:
            if api_version is None:
                raise Exception("api_version is required for Azure OpenAI API")
            if engine is None:
                return f'/openai/deployments?api-version={api_version}'
            else:
                return f'/openai/deployments/{quote_plus(engine)}{request_url}?api-version={api_version}'
        else:
            if engine is not None:
                return f'/engines/{quote_plus(engine)}{request_url}'
        return request_url

    @classmethod
    def get_chat_request_url(cls, api_type, api_version, engine):
        return cls.get_request_url('/chat/completions', api_type, api_version, engine)

    @classmethod
    def get_completions_request_url(cls, api_type, api_version, engine):
        return cls.get_request_url('/completions', api_type, api_version, engine)

    @classmethod
    def _chat_log_response_final(cls, logger, log_marks, kwargs, messages, start_time, role, content, err_msg=None):
        end_time = time.perf_counter()
        duration = end_time - start_time
        input_content = cls.converter.chat2raw(messages)
        if not err_msg:
            output_content = cls.converter.chat2raw([{'role': role, 'content': content}])
            log_result(logger, "Chat request", duration, log_marks, kwargs, input_content, output_content)
        else:
            log_exception(logger, "Chat request", duration, log_marks, kwargs, input_content, err_msg)

    @classmethod
    def _chat_log_response(cls, logger, log_marks, kwargs, messages, start_time, response, stream):
        if logger is not None:
            if stream:
                if inspect.isasyncgen(response):
                    async def wrapper(response):
                        content = ''
                        role = ''
                        async for data in response:
                            try:
                                message = data['choices'][0]['delta']
                                if 'role' in message:
                                    role = message['role']
                                if 'content' in message:
                                    content += message['content']
                            except (KeyError, IndexError):
                                pass
                            yield data
                        cls._chat_log_response_final(logger, log_marks, kwargs, messages, start_time, role, content)
                elif inspect.isgenerator(response):
                    def wrapper(response):
                        content = ''
                        role = ''
                        for data in response:
                            try:
                                message = data['choices'][0]['delta']
                                if 'role' in message:
                                    role = message['role']
                                if 'content' in message:
                                    content += message['content']
                            except (KeyError, IndexError):
                                pass
                            yield data
                        cls._chat_log_response_final(logger, log_marks, kwargs, messages, start_time, role, content)
                else:
                    raise Exception("response is not a generator or async generator in stream mode")
                response = wrapper(response)
            else:
                role = content = err_msg = None
                try:
                    role = response['choices'][0]['message']['role']
                    content = response['choices'][0]['message']['content']
                except (KeyError, IndexError):
                    err_msg = "Wrong response format, no message found"
                cls._chat_log_response_final(logger, log_marks, kwargs, messages, start_time, role, content, err_msg)
        return response
    
    @classmethod
    def _chat_log_exception(cls, logger, log_marks, kwargs, messages, start_time, exception: Exception):
        if logger is not None:
            end_time = time.perf_counter()
            duration = end_time - start_time
            input_content = cls.converter.chat2raw(messages)
            err_msg = exception2err_msg(exception)
            log_exception(logger, "Chat request", duration, log_marks, kwargs, input_content, err_msg)

    @classmethod
    def _completions_log_response_final(cls, logger, log_marks, kwargs, prompt, start_time, text, err_msg=None):
        end_time = time.perf_counter()
        duration = end_time - start_time
        input_content = prompt
        if not err_msg:
            output_content = text
            log_result(logger, "Completions request", duration, log_marks, kwargs, input_content, output_content)
        else:
            log_exception(logger, "Completions request", duration, log_marks, kwargs, input_content, err_msg)
    
    @classmethod
    def _completions_log_response(cls, logger, log_marks, kwargs, prompt, start_time, response, stream):
        if logger is not None:
            if stream:
                if inspect.isasyncgen(response):
                    async def wrapper(response):
                        text = ''
                        async for data in response:
                            try:
                                text += data['choices'][0]['text']
                            except (KeyError, IndexError):
                                pass
                            yield data
                        cls._completions_log_response_final(logger, log_marks, kwargs, prompt, start_time, text)
                elif inspect.isgenerator(response):
                    def wrapper(response):
                        text = ''
                        for data in response:
                            try:
                                text += data['choices'][0]['text']
                            except (KeyError, IndexError):
                                pass
                            yield data
                        cls._completions_log_response_final(logger, log_marks, kwargs, prompt, start_time, text)
                else:
                    raise Exception("response is not a generator or async generator in stream mode")
                response = wrapper(response)
            else:
                text = err_msg = None
                try:
                    text = response['choices'][0]['text']
                except (KeyError, IndexError):
                    err_msg = "Wrong response format, no text found"
                cls._completions_log_response_final(logger, log_marks, kwargs, prompt, start_time, text, err_msg)
        return response

    @classmethod
    def _completions_log_exception(cls, logger, log_marks, kwargs, prompt, start_time, exception: Exception):
        if logger is not None:
            end_time = time.perf_counter()
            duration = end_time - start_time
            input_content = prompt
            err_msg = exception2err_msg(exception)
            log_exception(logger, "Completions request", duration, log_marks, kwargs, input_content, err_msg)

    @classmethod
    def edits(cls, **kwargs):
        request_url = '/edits'
        return cls.api_request_endpoint(request_url, method='post', **kwargs)

    @classmethod
    def embeddings(cls, **kwargs):
        api_key, organization, api_base, api_type, api_version, engine, dest_url = cls.consume_kwargs(kwargs)
        request_url = cls.get_request_url('/embeddings', api_type, api_version, engine)
        return cls.api_request_endpoint(
            request_url, 
            method='post', 
            api_key=api_key,
            organization=organization,
            api_base=api_base,
            api_type=api_type,
            dest_url=dest_url,
            **kwargs
            )

    @classmethod
    def models_list(cls, **kwargs):
        api_key, organization, api_base, api_type, api_version, engine, dest_url = cls.consume_kwargs(kwargs)
        request_url = cls.get_request_url('/models', api_type, api_version, engine)
        return cls.api_request_endpoint(
            request_url, 
            method='get', 
            api_key=api_key,
            organization=organization,
            api_base=api_base,
            api_type=api_type,
            dest_url=dest_url,
            **kwargs
            )

    @classmethod
    def models_retrieve(cls, model, **kwargs):
        request_url = f'/models/{model}'
        return cls.api_request_endpoint(request_url, method='get', **kwargs)

    @classmethod
    def moderations(cls, **kwargs):
        request_url = '/moderations'
        return cls.api_request_endpoint(request_url, method='post', **kwargs)

    @classmethod
    def images_generations(cls, **kwargs):
        api_key, organization, api_base, api_type, api_version, engine, dest_url = cls.consume_kwargs(kwargs)
        if api_type and api_type.lower() in _API_TYPES_AZURE:
            request_url = f'/openai/images/generations:submit?api-version={api_version}'
            raw_response = True
        else:
            request_url = '/images/generations'
            raw_response = False
        response = cls.api_request_endpoint(
            request_url, 
            method='post', 
            api_key=api_key,
            organization=organization,
            api_base=api_base,
            api_type=api_type,
            raw_response=raw_response,
            dest_url=dest_url,
            **kwargs
            )
        if api_type and api_type.lower() in _API_TYPES_AZURE:
            # Azure image generation
            # use raw response to get poll_url
            poll_url = response.headers['operation-location']
            headers= { "api-key": api_key, "Content-Type": "application/json" }
            response = poll(
                url=poll_url, 
                method='get', 
                until=lambda response: response.json()['status'] == 'succeeded',
                failed=cls.check_image_failure,
                interval=lambda response: cls.get_retry(response) or 1,
                headers=headers, 
                ).json()
            return response.get('result', response)
        else:
            return response

    @staticmethod
    def check_image_failure(response):
        response_dict = response.json()
        if response_dict['status'] == 'failed':
            raise Exception(f"Image generation failed: {response_dict['error']['code']} {response_dict['error']['message']}")
    
    @staticmethod
    def get_retry(response):
        try:
            return int(response.headers.get('retry-after'))
        except:
            return None

    @classmethod
    def images_edits(cls, image, mask=None, **kwargs):
        request_url = '/images/edits'
        files = { 'image': image }
        if mask:
            files['mask'] = mask
        return cls.api_request_endpoint(request_url, method='post', files=files, **kwargs)

    @classmethod
    def images_variations(cls, image, **kwargs):
        request_url = '/images/variations'
        files = { 'image': image }
        return cls.api_request_endpoint(request_url, method='post', files=files, **kwargs)

    @classmethod
    def audio_transcriptions(cls, file, **kwargs):
        request_url = '/audio/transcriptions'
        files = { 'file': file }
        return cls.api_request_endpoint(request_url, method='post', files=files, **kwargs)

    @classmethod
    def audio_translations(cls, file, **kwargs):
        request_url = '/audio/translations'
        files = { 'file': file }
        return cls.api_request_endpoint(request_url, method='post', files=files, **kwargs)

    @classmethod
    def files_list(cls, **kwargs):
        request_url = '/files'
        return cls.api_request_endpoint(request_url, method='get', **kwargs)

    @classmethod
    def files_upload(cls, file, **kwargs):
        request_url = '/files'
        files = { 'file': file }
        return cls.api_request_endpoint(request_url, method='post', files=files, **kwargs)

    @classmethod
    def files_delete(cls, file_id, **kwargs):
        request_url = f'/files/{file_id}'
        return cls.api_request_endpoint(request_url, method='delete', **kwargs)

    @classmethod
    def files_retrieve(cls, file_id, **kwargs):
        request_url = f'/files/{file_id}'
        return cls.api_request_endpoint(request_url, method='get', **kwargs)

    @classmethod
    def files_retrieve_content(cls, file_id, **kwargs):
        request_url = f'/files/{file_id}/content'
        return cls.api_request_endpoint(request_url, method='get', **kwargs)

    @classmethod
    def finetunes_create(cls, **kwargs):
        request_url = '/fine-tunes'
        return cls.api_request_endpoint(request_url, method='post', **kwargs)

    @classmethod
    def finetunes_list(cls, **kwargs):
        request_url = '/fine-tunes'
        return cls.api_request_endpoint(request_url, method='get', **kwargs)

    @classmethod
    def finetunes_retrieve(cls, fine_tune_id, **kwargs):
        request_url = f'/fine-tunes/{fine_tune_id}'
        return cls.api_request_endpoint(request_url, method='get', **kwargs)

    @classmethod
    def finetunes_cancel(cls, fine_tune_id, **kwargs):
        request_url = f'/fine-tunes/{fine_tune_id}/cancel'
        return cls.api_request_endpoint(request_url, method='post', **kwargs)

    @classmethod
    def finetunes_list_events(cls, fine_tune_id, **kwargs):
        request_url = f'/fine-tunes/{fine_tune_id}/events'
        return cls.api_request_endpoint(request_url, method='get', **kwargs)

    @classmethod
    def finetunes_delete_model(cls, model, **kwargs):
        request_url = f'/models/{model}'
        return cls.api_request_endpoint(request_url, method='delete', **kwargs)

