"""
Deprecated OpenAIAPI code. It is not used in the current version of the 
library and is kept here for reference.
"""
import os
import time
import json

from ..endpoint_manager import Endpoint, EndpointManager
from .api_request import api_request, poll
from .._utils import get_request_url, join_url, _chat_log_response, _chat_log_exception, _completions_log_response, _completions_log_exception
from .. import utils

from .._constants import _API_BASE_OPENAI, _API_TYPE_OPENAI, _API_TYPES_AZURE


class OpenAIAPI:
    
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
    # cannot be None if using Azure API.
    api_version = None
    
    # set this to your model-engine map;
    # or environment variable MODEL_ENGINE_MAP will be used;
    # can be None.
    model_engine_map = None
    
    stream_chat = staticmethod(utils.stream_chat)
    stream_completions = staticmethod(utils.stream_completions)
    stream_chat_with_role = staticmethod(utils.stream_chat_with_role)
    
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
    def get_api_type(cls, api_type=None):
        return api_type or cls.api_type or os.environ.get('OPENAI_API_TYPE') or _API_TYPE_OPENAI
    
    @classmethod
    def get_api_version(cls, api_version=None):
        return api_version or cls.api_version or os.environ.get('OPENAI_API_VERSION')

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
        api_type = cls.get_api_type(kwargs.pop('api_type', api_type))
        api_version = cls.get_api_version(kwargs.pop('api_version', api_version))
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
    def chat(cls, messages, logger=None, log_marks=[], **kwargs):
        api_key, organization, api_base, api_type, api_version, engine, dest_url = cls.consume_kwargs(kwargs)
        start_time = time.perf_counter()
        try:
            response = cls.api_request_endpoint(
                get_request_url('/chat/completions', api_type, api_version, engine), 
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
            response = _chat_log_response(logger, log_marks, kwargs, messages, start_time, response, stream)
            return response
        except Exception as e:
            _chat_log_exception(logger, log_marks, kwargs, messages, start_time, e)
            raise e

    
    @classmethod
    def completions(cls, prompt, logger=None, log_marks=[], **kwargs):
        api_key, organization, api_base, api_type, api_version, engine, dest_url = cls.consume_kwargs(kwargs)
        start_time = time.perf_counter()
        try:
            response = cls.api_request_endpoint(
                get_request_url('/completions', api_type, api_version, engine), 
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
            response = _completions_log_response(logger, log_marks, kwargs, prompt, start_time, response, stream)
            return response
        except Exception as e:
            _completions_log_exception(logger, log_marks, kwargs, prompt, start_time, e)
            raise e

    
    @classmethod
    def edits(cls, **kwargs):
        request_url = '/edits'
        return cls.api_request_endpoint(request_url, method='post', **kwargs)

    @classmethod
    def embeddings(cls, **kwargs):
        api_key, organization, api_base, api_type, api_version, engine, dest_url = cls.consume_kwargs(kwargs)
        request_url = get_request_url('/embeddings', api_type, api_version, engine)
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
        request_url = get_request_url('/models', api_type, api_version, engine)
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

