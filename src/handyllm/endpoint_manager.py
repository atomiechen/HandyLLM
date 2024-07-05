__all__ = [
    "Endpoint",
    "EndpointManager",
]

from threading import Lock
from collections.abc import MutableSequence
from typing import Iterable, Mapping, Optional, Union
import yaml

from .types import PathType
from ._utils import isiterable


class Endpoint:
    def __init__(
        self, 
        name=None,
        api_key=None, 
        organization=None, 
        api_base=None, 
        api_type=None,
        api_version=None, 
        model_engine_map=None, 
        dest_url=None, 
        ):
        self.name = name if name else f"ep_{id(self)}"
        self.api_key = api_key
        self.organization = organization
        self.api_base = api_base
        self.api_type = api_type
        self.api_version = api_version
        self.model_engine_map = model_engine_map if model_engine_map else {}
        self.dest_url = dest_url

    def __str__(self) -> str:
        # do not print api_key
        listed_attributes = [
            f'name={repr(self.name)}' if self.name else None,
            f'api_key=*' if self.api_key else None,
            f'organization={repr(self.organization)}' if self.organization else None,
            f'api_base={repr(self.api_base)}' if self.api_base else None,
            f'api_type={repr(self.api_type)}' if self.api_type else None,
            f'api_version={repr(self.api_version)}' if self.api_version else None,
            f'model_engine_map={repr(self.model_engine_map)}' if self.model_engine_map else None,
            f'dest_url={repr(self.dest_url)}' if self.dest_url else None,
        ]
        # remove None in listed_attributes
        listed_attributes = [item for item in listed_attributes if item]
        return f"Endpoint({', '.join(listed_attributes)})"

    def get_api_info(self):
        return (
            self.api_key, 
            self.organization, 
            self.api_base, 
            self.api_type, 
            self.api_version, 
            self.model_engine_map, 
            self.dest_url, 
        )


class EndpointManager(MutableSequence):

    def __init__(self, endpoints: Optional[Iterable] = None, load_path: Optional[PathType] = None):
        self._lock = Lock()
        self._last_idx_endpoint = 0
        self._endpoints = []
        if endpoints is not None:
            self.load_from_list(endpoints, override=False)
        if load_path is not None:
            self.load_from(load_path, override=False)

    def clear(self):
        self._last_idx_endpoint = 0
        self._endpoints.clear()
        
    def __len__(self) -> int:
        return len(self._endpoints)

    def __getitem__(self, idx):
        return self._endpoints[idx]

    def __setitem__(self, idx, endpoint):
        self._endpoints[idx] = endpoint

    def __delitem__(self, idx):
        del self._endpoints[idx]

    def insert(self, index: int, value: Endpoint):
        self._endpoints.insert(index, value)

    def add_endpoint_by_info(self, **kwargs):
        endpoint = Endpoint(**kwargs)
        self.append(endpoint)

    def get_next_endpoint(self) -> Endpoint:
        if len(self._endpoints) == 0:
            raise ValueError("No endpoint available")
        with self._lock:
            endpoint = self._endpoints[self._last_idx_endpoint]
            if self._last_idx_endpoint == len(self._endpoints) - 1:
                self._last_idx_endpoint = 0
            else:
                self._last_idx_endpoint += 1
            return endpoint

    def load_from_list(self, obj: Iterable[Union[Mapping, Endpoint]], override=False):
        if not isiterable(obj):
            raise ValueError("obj must be a non-str iterable (list, tuple, etc.)")
        if override:
            self.clear()
        for ep in obj:
            if isinstance(ep, Endpoint):
                self.append(ep)
            elif isinstance(ep, Mapping):
                self.add_endpoint_by_info(**ep)
            else:
                raise ValueError(f"Unsupported type {type(ep)}")

    def load_from(self, path: PathType, encoding="utf-8", override=False):
        with open(path, "r", encoding=encoding) as fin:
            obj = yaml.safe_load(fin)
        if isinstance(obj, Mapping):
            obj = obj.get("endpoints", None)
        if obj:
            self.load_from_list(obj, override=override)

