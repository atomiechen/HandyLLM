"""
legacy OpenAIAPI support
"""

from typing import Union
from .openai_client import OpenAIClient, ClientMode
from . import utils


_module_client: Union[OpenAIClient, None] = None

def _load_client() -> OpenAIClient:
    global _module_client
    if _module_client is None:
        _module_client = OpenAIClient(
            ClientMode.SYNC,
            api_base=OpenAIAPI.api_base,
            api_key=OpenAIAPI.api_key,
            organization=OpenAIAPI.organization,
            api_type=OpenAIAPI.api_type,
            api_version=OpenAIAPI.api_version,
            model_engine_map=OpenAIAPI.model_engine_map,
            )
    return _module_client


class _OpenAIAPIMeta(type):
    def __getattr__(cls, name):
        def api_method(*args, **kwargs):
            return getattr(_load_client(), name)(*args, **kwargs).call()
        return api_method


class OpenAIAPI(metaclass=_OpenAIAPIMeta):
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

    stream_chat = utils.stream_chat
    stream_completions = utils.stream_completions
    stream_chat_with_role = utils.stream_chat_with_role

