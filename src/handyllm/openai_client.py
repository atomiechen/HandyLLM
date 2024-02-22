import os
import json
import time
from enum import Enum, auto
import asyncio
from typing import Union
import requests
import httpx

from .endpoint_manager import Endpoint, EndpointManager
from .requestor import Requestor
from ._utils import get_request_url, join_url, _chat_log_response, _chat_log_exception, _completions_log_response, _completions_log_exception
from . import _API_BASE_OPENAI, _API_TYPE_OPENAI, _API_TYPES_AZURE


class ClientMode(Enum):
    SYNC = auto()
    ASYNC = auto()
    BOTH = auto()


class OpenAIClient:
    def __init__(
        self, 
        mode: Union[str, ClientMode] = ClientMode.SYNC,
        *,
        base_url=None,
        api_base=None,
        api_key=None,
        organization=None,
        api_type=None,
        api_version=None,
        model_engine_map=None,
        ) -> None:
        # convert string to enum
        if isinstance(mode, str):
            mode = mode.upper()
            if mode not in ClientMode.__members__:
                raise ValueError("Invalid client mode specified")
            mode = ClientMode[mode]
        elif not isinstance(mode, ClientMode):
            raise TypeError("Invalid client mode specified")

        if mode == ClientMode.SYNC or mode == ClientMode.BOTH:
            self.sync_client = requests.Session()
        else:
            self.sync_client = None
        
        if mode == ClientMode.ASYNC or mode == ClientMode.BOTH:
            self.async_client = httpx.AsyncClient()
        else:
            self.async_client = None
        
        # set this to your API base;
        # or environment variable OPENAI_API_BASE will be used.
        # can be None (roll back to default).
        self.api_base = base_url or api_base
        
        # set this to your API key; 
        # or environment variable OPENAI_API_KEY will be used.
        self.api_key = api_key
        
        # set this to your organization ID; 
        # or environment variable OPENAI_ORGANIZATION will be used;
        # can be None.
        self.organization = organization
        
        # set this to your API type;
        # or environment variable OPENAI_API_TYPE will be used;
        # can be None (roll back to default).
        self.api_type = api_type
        
        # set this to your API version;
        # or environment variable OPENAI_API_VERSION will be used;
        # can be None.
        self.api_version = api_version
        
        # set this to your model-engine map;
        # or environment variable MODEL_ENGINE_MAP will be used;
        # can be None.
        self.model_engine_map = model_engine_map
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, *args):
        await self.aclose()
    
    def __del__(self):
        self.close()
    
    def close(self):
        if self.async_client is not None:
            try:
                asyncio.get_running_loop().create_task(self.async_client.aclose())
                self.async_client = None
            except Exception:
                pass
        if self.sync_client is not None:
            try:
                self.sync_client.close()
                self.sync_client = None
            except Exception:
                pass
    
    async def aclose(self):
        if self.async_client is not None:
            try:
                await self.async_client.aclose()
                self.async_client = None
            except Exception:
                pass
        if self.sync_client is not None:
            try:
                self.sync_client.close()
                self.sync_client = None
            except Exception:
                pass

    def _infer_api_key(self, api_key=None):
        return api_key or self.api_key or os.environ.get('OPENAI_API_KEY')
    
    def _infer_organization(self, organization=None):
        return organization or self.organization or os.environ.get('OPENAI_ORGANIZATION')
    
    def _infer_api_base(self, api_base=None):
        return api_base or self.api_base or os.environ.get('OPENAI_API_BASE') or _API_BASE_OPENAI
    
    def _infer_api_type(self, api_type=None):
        return api_type or self.api_type or os.environ.get('OPENAI_API_TYPE') or _API_TYPE_OPENAI

    def _infer_api_version(self, api_version=None):
        return api_version or self.api_version or os.environ.get('OPENAI_API_VERSION')

    def _infer_model_engine_map(self, model_engine_map=None):
        if model_engine_map:
            return model_engine_map
        if self.model_engine_map:
            return self.model_engine_map
        try:
            json_str = os.environ.get('MODEL_ENGINE_MAP')
            return json.loads(json_str)
        except:
            return None

    def _consume_kwargs(self, kwargs):
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
        api_key = self._infer_api_key(kwargs.pop('api_key', api_key))
        organization = self._infer_organization(kwargs.pop('organization', organization))
        api_base = self._infer_api_base(kwargs.pop('api_base', api_base))
        api_type = self._infer_api_type(kwargs.pop('api_type', api_type))
        api_version = self._infer_api_version(kwargs.pop('api_version', api_version))
        model_engine_map = self._infer_model_engine_map(kwargs.pop('model_engine_map', model_engine_map))

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

    def _make_requestor(self, request_url, **kwargs) -> Requestor:
        api_key, organization, api_base, api_type, api_version, engine, dest_url = self._consume_kwargs(kwargs)
        url = join_url(api_base, request_url)
        requestor = Requestor(api_type, url, api_key, organization=organization, dest_url=dest_url, **kwargs)
        requestor.set_sync_client(self.sync_client)
        requestor.set_async_client(self.async_client)
        return requestor

    def chat(self, messages, logger=None, log_marks=[], **kwargs) -> Requestor:
        api_key, organization, api_base, api_type, api_version, engine, dest_url = self._consume_kwargs(kwargs)
        requestor = self._make_requestor(
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
        requestor.set_prepare_callback(
            lambda *args : time.perf_counter()  # start_time
        )
        stream = kwargs.get('stream', False)
        requestor.set_response_callback(
            lambda response, start_time: _chat_log_response(logger, log_marks, kwargs, messages, start_time, response, stream)
        )
        requestor.set_exception_callback(
            lambda exception, start_time: _chat_log_exception(logger, log_marks, kwargs, messages, start_time, exception)
        )
        return requestor

    def completions(self, prompt, logger=None, log_marks=[], **kwargs) -> Requestor:
        api_key, organization, api_base, api_type, api_version, engine, dest_url = self._consume_kwargs(kwargs)
        requestor = self._make_requestor(
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
        requestor.set_prepare_callback(
            lambda *args : time.perf_counter()  # start_time
        )
        stream = kwargs.get('stream', False)
        requestor.set_response_callback(
            lambda response, start_time: _completions_log_response(logger, log_marks, kwargs, prompt, start_time, response, stream)
        )
        requestor.set_exception_callback(
            lambda exception, start_time: _completions_log_exception(logger, log_marks, kwargs, prompt, start_time, exception)
        )
        return requestor

    def edits(self, **kwargs) -> Requestor:
        return self._make_requestor('/edits', method='post', **kwargs)

    def embeddings(self, **kwargs) -> Requestor:
        api_key, organization, api_base, api_type, api_version, engine, dest_url = self._consume_kwargs(kwargs)
        return self._make_requestor(
            get_request_url('/embeddings', api_type, api_version, engine), 
            method='post', 
            api_key=api_key,
            organization=organization,
            api_base=api_base,
            api_type=api_type,
            dest_url=dest_url,
            **kwargs
            )

    def models_list(self, **kwargs) -> Requestor:
        api_key, organization, api_base, api_type, api_version, engine, dest_url = self._consume_kwargs(kwargs)
        return self._make_requestor(
            get_request_url('/models', api_type, api_version, engine), 
            method='get', 
            api_key=api_key,
            organization=organization,
            api_base=api_base,
            api_type=api_type,
            dest_url=dest_url,
            **kwargs
            )

    def models_retrieve(self, model, **kwargs) -> Requestor:
        return self._make_requestor(f'/models/{model}', method='get', **kwargs)

    def moderations(self, **kwargs) -> Requestor:
        return self._make_requestor('/moderations', method='post', **kwargs)

    def images_generations(self, **kwargs) -> Requestor:
        api_key, organization, api_base, api_type, api_version, engine, dest_url = self._consume_kwargs(kwargs)
        if api_type and api_type.lower() in _API_TYPES_AZURE and api_version in [
            "2023-06-01-preview",
            "2023-07-01-preview",
            "2023-08-01-preview",
            "2023-09-01-preview",
            "2023-10-01-preview",
        ]:
            # Azure image generation DALL-E 2
            # use raw response to get poll_url
            request_url = f'/openai/images/generations:submit?api-version={api_version}'
            azure_poll=True
        else:
            # OpenAI image generation, or Azure image generation DALL-E 3 and newer
            request_url = get_request_url('/images/generations', api_type, api_version, engine)
            azure_poll=False
        return self._make_requestor(
            request_url, 
            method='post', 
            api_key=api_key,
            organization=organization,
            api_base=api_base,
            api_type=api_type,
            dest_url=dest_url,
            azure_poll=azure_poll,
            **kwargs
            )

    def images_edits(self, image, mask=None, **kwargs) -> Requestor:
        files = { 'image': image }
        if mask:
            files['mask'] = mask
        return self._make_requestor('/images/edits', method='post', files=files, **kwargs)

    def images_variations(self, image, **kwargs) -> Requestor:
        files = { 'image': image }
        return self._make_requestor('/images/variations', method='post', files=files, **kwargs)

    def audio_transcriptions(self, file, **kwargs) -> Requestor:
        files = { 'file': file }
        return self._make_requestor('/audio/transcriptions', method='post', files=files, **kwargs)

    def audio_translations(self, file, **kwargs) -> Requestor:
        files = { 'file': file }
        return self._make_requestor('/audio/translations', method='post', files=files, **kwargs)

    def files_list(self, **kwargs) -> Requestor:
        return self._make_requestor('/files', method='get', **kwargs)

    def files_upload(self, file, **kwargs) -> Requestor:
        files = { 'file': file }
        return self._make_requestor('/files', method='post', files=files, **kwargs)

    def files_delete(self, file_id, **kwargs) -> Requestor:
        return self._make_requestor(f'/files/{file_id}', method='delete', **kwargs)

    def files_retrieve(self, file_id, **kwargs) -> Requestor:
        return self._make_requestor(f'/files/{file_id}', method='get', **kwargs)

    def files_retrieve_content(self, file_id, **kwargs) -> Requestor:
        return self._make_requestor(f'/files/{file_id}/content', method='get', **kwargs)

    def finetunes_create(self, **kwargs) -> Requestor:
        return self._make_requestor('/fine-tunes', method='post', **kwargs)

    def finetunes_list(self, **kwargs) -> Requestor:
        return self._make_requestor('/fine-tunes', method='get', **kwargs)

    def finetunes_retrieve(self, fine_tune_id, **kwargs) -> Requestor:
        return self._make_requestor(f'/fine-tunes/{fine_tune_id}', method='get', **kwargs)

    def finetunes_cancel(self, fine_tune_id, **kwargs) -> Requestor:
        return self._make_requestor(f'/fine-tunes/{fine_tune_id}/cancel', method='post', **kwargs)

    def finetunes_list_events(self, fine_tune_id, **kwargs) -> Requestor:
        return self._make_requestor(f'/fine-tunes/{fine_tune_id}/events', method='get', **kwargs)

    def finetunes_delete_model(self, model, **kwargs) -> Requestor:
        return self._make_requestor(f'/models/{model}', method='delete', **kwargs)

