"""
legacy OpenAIAPI support
"""

from typing import Union
from .openai_client import OpenAIClient, ClientMode
from .requestor import Requestor
from . import utils


class _ModuleClient(OpenAIClient):
    """An inherited class from OpenAIClient using OpenAIAPI's properties."""

    @property
    def api_key(self):
        return OpenAIAPI.api_key

    @api_key.setter
    def api_key(self, value):
        OpenAIAPI.api_key = value
    
    @property
    def organization(self):
        return OpenAIAPI.organization
    
    @organization.setter
    def organization(self, value):
        OpenAIAPI.organization = value
    
    @property
    def api_base(self):
        return OpenAIAPI.api_base
    
    @api_base.setter
    def api_base(self, value):
        OpenAIAPI.api_base = value
    
    @property
    def api_type(self):
        return OpenAIAPI.api_type
    
    @api_type.setter
    def api_type(self, value):
        OpenAIAPI.api_type = value
    
    @property
    def api_version(self):
        return OpenAIAPI.api_version
    
    @api_version.setter
    def api_version(self, value):
        OpenAIAPI.api_version = value
    
    @property
    def model_engine_map(self):
        return OpenAIAPI.model_engine_map
    
    @model_engine_map.setter
    def model_engine_map(self, value):
        OpenAIAPI.model_engine_map = value


_module_client: Union[_ModuleClient, None] = None

def _load_client() -> _ModuleClient:
    """Lazy load the module client."""
    global _module_client
    if _module_client is None:
        _module_client = _ModuleClient(
            ClientMode.SYNC,
            api_base=OpenAIAPI.api_base,
            api_key=OpenAIAPI.api_key,
            organization=OpenAIAPI.organization,
            api_type=OpenAIAPI.api_type,
            api_version=OpenAIAPI.api_version,
            model_engine_map=OpenAIAPI.model_engine_map,
            )
    return _module_client


class _OpenAIClientProxy:
    """A proxy for all OpenAIClient API methods."""

    # set this to your API type;
    # or environment variable OPENAI_API_TYPE will be used;
    # can be None (roll back to default).
    api_type: Union[str, None] = None

    # set this to your API base;
    # or environment variable OPENAI_API_BASE will be used.
    # can be None (roll back to default).
    api_base: Union[str, None] = None
    
    # set this to your API key; 
    # or environment variable OPENAI_API_KEY will be used.
    api_key: Union[str, None] = None
    
    # set this to your organization ID; 
    # or environment variable OPENAI_ORGANIZATION / OPENAI_ORG_ID will be used;
    # can be None.
    organization: Union[str, None] = None
    
    # set this to your API version;
    # or environment variable OPENAI_API_VERSION will be used;
    # cannot be None if using Azure API.
    api_version: Union[str, None] = None
    
    # set this to your model-engine map;
    # or environment variable MODEL_ENGINE_MAP will be used;
    # can be None.
    model_engine_map: Union[dict, None] = None

    stream_chat = staticmethod(utils.stream_chat)
    stream_completions = staticmethod(utils.stream_completions)
    stream_chat_with_role = staticmethod(utils.stream_chat_with_role)


    def __getattr__(self, name: str):
        """Catch all API methods."""
        try:
            method = getattr(_load_client(), name)
        except AttributeError:
            raise AttributeError(f"'OpenAIAPI' has no attribute '{name}'")
        if not getattr(method, "is_api", False):
            raise ValueError(f"{name} is not an API method")
        def modified_api_method(*args, **kwargs):
            requestor = method(*args, **kwargs)
            # api methods must return a requestor
            assert isinstance(requestor, Requestor)
            return requestor.call()
        return modified_api_method


OpenAIAPI = _OpenAIClientProxy()

