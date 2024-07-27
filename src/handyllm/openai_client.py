from __future__ import annotations
__all__ = [
    'OpenAIClient',
    'ClientMode',
]

from typing import Iterable, Mapping, Optional, TypeVar, Union
import os
import json
import time
from enum import Enum, auto
import asyncio

import yaml

from .endpoint_manager import Endpoint, EndpointManager
from .requestor import Requestor, BinRequestor, DictRequestor, ChatRequestor, CompletionsRequestor
from ._utils import get_request_url, join_url, _chat_log_response, _chat_log_exception, _completions_log_response, _completions_log_exception
from ._constants import API_BASE_OPENAI, API_TYPE_OPENAI, API_TYPES_AZURE, TYPE_API_TYPES
from .types import PathType


RequestorType = TypeVar('RequestorType', bound='Requestor')

def api(func):
    func.is_api = True
    return func


class ClientMode(Enum):
    SYNC = auto()
    ASYNC = auto()
    BOTH = auto()


class OpenAIClient:
    # set this to your API type;
    # or environment variable OPENAI_API_TYPE will be used;
    # can be None (roll back to default).
    api_type: Optional[TYPE_API_TYPES]

    # set this to your API base;
    # or environment variable OPENAI_API_BASE will be used.
    # can be None (roll back to default).
    api_base: Optional[str]
    
    # set this to your API key; 
    # or environment variable OPENAI_API_KEY will be used.
    api_key: Optional[str]
    
    # set this to your organization ID; 
    # or environment variable OPENAI_ORGANIZATION / OPENAI_ORG_ID will be used;
    # can be None.
    organization: Optional[str]
    
    # set this to your API version;
    # or environment variable OPENAI_API_VERSION will be used;
    # cannot be None if using Azure API.
    api_version: Optional[str]
    
    # set this to your model-engine map;
    # or environment variable MODEL_ENGINE_MAP will be used;
    # can be None.
    model_engine_map: Optional[dict[str, str]]
    
    # set this to your endpoint manager
    endpoint_manager: Optional[EndpointManager] = None
    
    def __init__(
        self, 
        mode: Union[str, ClientMode] = ClientMode.SYNC,
        *,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        api_type: Optional[TYPE_API_TYPES] = None,
        api_version: Optional[str] = None,
        model_engine_map: Optional[dict[str, str]] = None,
        endpoint_manager: Optional[EndpointManager] = None,
        endpoints: Optional[Iterable] = None,
        load_path: Optional[PathType] = None,
        ) -> None:
        self._sync_client = None
        self._async_client = None
        
        # convert string to enum
        if isinstance(mode, str):
            mode = mode.upper()
            if mode not in ClientMode.__members__:
                raise ValueError("Invalid client mode specified")
            mode = ClientMode[mode]
        elif not isinstance(mode, ClientMode):
            raise TypeError("Invalid client mode specified")
        
        if mode == ClientMode.SYNC or mode == ClientMode.BOTH:
            # lazy import
            import requests
            self._sync_client = requests.Session()
        
        if mode == ClientMode.ASYNC or mode == ClientMode.BOTH:
            # lazy import
            import httpx
            self._async_client = httpx.AsyncClient()
        
        self.api_base = api_base
        self.api_key = api_key
        self.organization = organization
        self.api_type = api_type
        self.api_version = api_version
        self.model_engine_map = model_engine_map
        
        if endpoint_manager:
            if not isinstance(endpoint_manager, EndpointManager):
                raise ValueError("endpoint_manager must be an instance of EndpointManager")
        elif endpoints:
            endpoint_manager = EndpointManager(endpoints=endpoints)
        self.endpoint_manager = endpoint_manager
        
        if load_path:
            self.load_from(load_path, override=False)
    
    def load_from(self, path: PathType, encoding="utf-8", override=False):
        with open(path, "r", encoding=encoding) as fin:
            obj = yaml.safe_load(fin)
        if obj:
            self.load_from_obj(obj, override=override)
    
    def load_from_obj(self, obj: Mapping, override=False):
        if not isinstance(obj, Mapping):
           raise ValueError("obj must be a mapping (dict, etc.)") 
        api_base = obj.get("api_base", None)
        api_key = obj.get("api_key", None)
        organization = obj.get("organization", None)
        api_type = obj.get("api_type", None)
        api_version = obj.get("api_version", None)
        model_engine_map = obj.get("model_engine_map", None)
        item = obj.get("endpoints", None)
        if api_base and (override or not self.api_base):
            self.api_base = api_base
        if api_key and (override or not self.api_key):
            self.api_key = api_key
        if organization and (override or not self.organization):
            self.organization = organization
        if api_type and (override or not self.api_type):
            self.api_type = api_type
        if api_version and (override or not self.api_version):
            self.api_version = api_version
        if model_engine_map and (override or not self.model_engine_map):
            self.model_engine_map = model_engine_map
        if item and (override or not self.endpoint_manager):
            if self.endpoint_manager is None:
                self.endpoint_manager = EndpointManager()
            self.endpoint_manager.load_from_list(item, override=override)
    
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
        if self._async_client is not None:
            try:
                asyncio.get_running_loop().create_task(self._async_client.aclose())
                self._async_client = None
            except Exception:
                pass
        if self._sync_client is not None:
            try:
                self._sync_client.close()
                self._sync_client = None
            except Exception:
                pass
    
    async def aclose(self):
        if self._async_client is not None:
            try:
                await self._async_client.aclose()
                self._async_client = None
            except Exception:
                pass
        if self._sync_client is not None:
            try:
                self._sync_client.close()
                self._sync_client = None
            except Exception:
                pass

    def _infer_api_key(self, api_key=None):
        return api_key or self.api_key or os.environ.get('OPENAI_API_KEY')
    
    def _infer_organization(self, organization=None):
        return organization or self.organization or os.environ.get('OPENAI_ORGANIZATION') or os.environ.get('OPENAI_ORG_ID')
    
    def _infer_api_base(self, api_base=None):
        return api_base or self.api_base or os.environ.get('OPENAI_API_BASE') or API_BASE_OPENAI
    
    def _infer_api_type(self, api_type=None):
        return (api_type or self.api_type or os.environ.get('OPENAI_API_TYPE') or API_TYPE_OPENAI).lower()

    def _infer_api_version(self, api_version=None):
        return api_version or self.api_version or os.environ.get('OPENAI_API_VERSION')

    def _infer_model_engine_map(self, model_engine_map=None):
        if model_engine_map:
            return model_engine_map
        if self.model_engine_map:
            return self.model_engine_map
        json_str = os.environ.get('MODEL_ENGINE_MAP')
        if not json_str:
            return None
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None

    def _consume_kwargs(self, kwargs):
        api_key = organization = api_base = api_type = api_version = engine = model_engine_map = dest_url = endpoint_manager = None

        # read API info from endpoint_manager
        endpoints = kwargs.pop('endpoints', None)
        if endpoints:
            endpoint_manager = EndpointManager(endpoints=endpoints)
        endpoint_manager = kwargs.pop('endpoint_manager', endpoint_manager) or self.endpoint_manager
        if endpoint_manager is not None and not kwargs.get('__endpoint_manager_used__', False):
            if not isinstance(endpoint_manager, EndpointManager):
                raise Exception("endpoint_manager must be an instance of EndpointManager")
            # get_next_endpoint() will be called once for each request
            api_key, organization, api_base, api_type, api_version, model_engine_map, dest_url = endpoint_manager.get_next_endpoint().get_api_info()
            kwargs['__endpoint_manager_used__'] = True

        # read API info from endpoint (override API info from endpoint_manager)
        endpoint = kwargs.pop('endpoint', None)
        if endpoint is not None:
            if not isinstance(endpoint, Endpoint):
                endpoint = Endpoint(**endpoint)
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
        if api_type and api_type in API_TYPES_AZURE:
            if not engine:
                # keep or consume model parameter
                keep_model = kwargs.pop('keep_model', False)
                if keep_model:
                    model = kwargs.get('model', None)
                else:
                    model = kwargs.pop('model', None)
                if model:
                    if model_engine_map:
                        engine = model_engine_map.get(model, model)
                    else:
                        engine = model
        dest_url = kwargs.pop('dest_url', dest_url)
        return api_key, organization, api_base, api_type, api_version, engine, dest_url

    def _make_requestor(
        self, 
        request_url: str, 
        requestor_cls: type[RequestorType],
        **kwargs
        ):
        api_key, organization, api_base, api_type, api_version, engine, dest_url = self._consume_kwargs(kwargs)
        url = join_url(api_base, request_url)
        filtered_kwargs = {k: v for k, v in kwargs.items() if not k.startswith('_')}
        requestor = requestor_cls(api_type, url, api_key, organization=organization, dest_url=dest_url, **filtered_kwargs)
        requestor.set_sync_client(self._sync_client)
        requestor.set_async_client(self._async_client)
        return requestor

    def _make_dict_requestor(self, request_url, **kwargs):
        return self._make_requestor(request_url, requestor_cls=DictRequestor, **kwargs)

    def _make_bin_requestor(self, request_url, **kwargs):
        return self._make_requestor(request_url, requestor_cls=BinRequestor, **kwargs)

    @api
    def chat(self, messages, logger=None, log_marks=[], **kwargs):
        api_key, organization, api_base, api_type, api_version, engine, dest_url = self._consume_kwargs(kwargs)
        requestor = self._make_requestor(
            get_request_url('/chat/completions', api_type, api_version, engine), 
            requestor_cls=ChatRequestor,
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

    @api
    def completions(self, prompt, logger=None, log_marks=[], **kwargs):
        api_key, organization, api_base, api_type, api_version, engine, dest_url = self._consume_kwargs(kwargs)
        requestor = self._make_requestor(
            get_request_url('/completions', api_type, api_version, engine), 
            requestor_cls=CompletionsRequestor,
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

    @api
    def edits(self, **kwargs):
        return self._make_dict_requestor('/edits', method='post', **kwargs)

    @api
    def embeddings(self, **kwargs):
        api_key, organization, api_base, api_type, api_version, engine, dest_url = self._consume_kwargs(kwargs)
        return self._make_dict_requestor(
            get_request_url('/embeddings', api_type, api_version, engine), 
            method='post', 
            api_key=api_key,
            organization=organization,
            api_base=api_base,
            api_type=api_type,
            dest_url=dest_url,
            **kwargs
            )

    @api
    def models_list(self, **kwargs):
        api_key, organization, api_base, api_type, api_version, engine, dest_url = self._consume_kwargs(kwargs)
        return self._make_dict_requestor(
            get_request_url('/models', api_type, api_version, engine), 
            method='get', 
            api_key=api_key,
            organization=organization,
            api_base=api_base,
            api_type=api_type,
            dest_url=dest_url,
            **kwargs
            )

    @api
    def models_retrieve(self, model, **kwargs):
        return self._make_dict_requestor(f'/models/{model}', method='get', **kwargs)

    @api
    def moderations(self, **kwargs):
        return self._make_dict_requestor('/moderations', method='post', **kwargs)

    @api
    def images_generations(self, **kwargs):
        api_key, organization, api_base, api_type, api_version, engine, dest_url = self._consume_kwargs(kwargs)
        if api_type and api_type in API_TYPES_AZURE and api_version in [
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
        return self._make_dict_requestor(
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

    @api
    def images_edits(self, image, mask=None, **kwargs):
        files = { 'image': image }
        if mask:
            files['mask'] = mask
        return self._make_dict_requestor('/images/edits', method='post', files=files, **kwargs)

    @api
    def images_variations(self, image, **kwargs):
        files = { 'image': image }
        return self._make_dict_requestor('/images/variations', method='post', files=files, **kwargs)

    @api
    def audio_speech(self, stream=False, chunk_size=1024, **kwargs):
        api_key, organization, api_base, api_type, api_version, engine, dest_url = self._consume_kwargs(kwargs)
        # NOTE: this api needs both model and engine parameters
        return self._make_bin_requestor(
            get_request_url('/audio/speech', api_type, api_version, engine),
            method='post', 
            api_key=api_key,
            organization=organization,
            api_base=api_base,
            api_type=api_type,
            dest_url=dest_url,
            stream=stream, 
            chunk_size=chunk_size, 
            # avoid poping model parameter
            keep_model=True,
            **kwargs
            )

    @api
    def audio_transcriptions(self, file, **kwargs):
        files = { 'file': file }
        api_key, organization, api_base, api_type, api_version, engine, dest_url = self._consume_kwargs(kwargs)
        return self._make_dict_requestor(
            get_request_url('/audio/transcriptions', api_type, api_version, engine),
            method='post', 
            api_key=api_key,
            organization=organization,
            api_base=api_base,
            api_type=api_type,
            dest_url=dest_url,
            files=files, 
            **kwargs
            )

    @api
    def audio_translations(self, file, **kwargs):
        files = { 'file': file }
        return self._make_dict_requestor('/audio/translations', method='post', files=files, **kwargs)

    @api
    def files_list(self, **kwargs):
        return self._make_dict_requestor('/files', method='get', **kwargs)

    @api
    def files_upload(self, file, **kwargs):
        files = { 'file': file }
        return self._make_dict_requestor('/files', method='post', files=files, **kwargs)

    @api
    def files_delete(self, file_id, **kwargs):
        return self._make_dict_requestor(f'/files/{file_id}', method='delete', **kwargs)

    @api
    def files_retrieve(self, file_id, **kwargs):
        return self._make_dict_requestor(f'/files/{file_id}', method='get', **kwargs)

    @api
    def files_retrieve_content(self, file_id, **kwargs):
        return self._make_dict_requestor(f'/files/{file_id}/content', method='get', **kwargs)

    @api
    def finetunes_create(self, **kwargs):
        return self._make_dict_requestor('/fine-tunes', method='post', **kwargs)

    @api
    def finetunes_list(self, **kwargs):
        return self._make_dict_requestor('/fine-tunes', method='get', **kwargs)

    @api
    def finetunes_retrieve(self, fine_tune_id, **kwargs):
        return self._make_dict_requestor(f'/fine-tunes/{fine_tune_id}', method='get', **kwargs)

    @api
    def finetunes_cancel(self, fine_tune_id, **kwargs):
        return self._make_dict_requestor(f'/fine-tunes/{fine_tune_id}/cancel', method='post', **kwargs)

    @api
    def finetunes_list_events(self, fine_tune_id, **kwargs):
        return self._make_dict_requestor(f'/fine-tunes/{fine_tune_id}/events', method='get', **kwargs)

    @api
    def finetunes_delete_model(self, model, **kwargs):
        return self._make_dict_requestor(f'/models/{model}', method='delete', **kwargs)

