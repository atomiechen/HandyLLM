"""
legacy OpenAIAPI support
"""

__all__ = [
    "OpenAIAPI"
]

from typing import Union
from .openai_client import OpenAIClient, ClientMode
from .requestor import Requestor
from . import utils


_module_client: Union[OpenAIClient, None] = None

def _load_client() -> OpenAIClient:
    """Lazy load the module client."""
    global _module_client
    if _module_client is None:
        _module_client = OpenAIClient(ClientMode.SYNC)
    return _module_client


class _OpenAIClientProxy:
    """A proxy for all OpenAIClient API methods."""
    
    @property
    def api_key(self):
        return _load_client().api_key

    @api_key.setter
    def api_key(self, value):
        _load_client().api_key = value
    
    @property
    def organization(self):
        return _load_client().organization
    
    @organization.setter
    def organization(self, value):
        _load_client().organization = value
    
    @property
    def api_base(self):
        return _load_client().api_base
    
    @api_base.setter
    def api_base(self, value):
        _load_client().api_base = value
    
    @property
    def api_type(self):
        return _load_client().api_type
    
    @api_type.setter
    def api_type(self, value):
        _load_client().api_type = value
    
    @property
    def api_version(self):
        return _load_client().api_version
    
    @api_version.setter
    def api_version(self, value):
        _load_client().api_version = value
    
    @property
    def model_engine_map(self):
        return _load_client().model_engine_map
    
    @model_engine_map.setter
    def model_engine_map(self, value):
        _load_client().model_engine_map = value

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

