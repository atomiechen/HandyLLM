from __future__ import annotations

__all__ = [
    "Endpoint",
    "EndpointManager",
]

import os
from json import JSONDecodeError
from threading import Lock
from collections.abc import MutableSequence
from typing import Iterable, Mapping, Optional, Union, cast

from .types import PathType
from ._utils import isiterable
from ._io import yaml_load, json_loads
from ._constants import TYPE_API_TYPES


class Endpoint:
    def __init__(
        self,
        name: Optional[str] = None,
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        api_base: Optional[str] = None,
        api_type: Optional[TYPE_API_TYPES] = None,
        api_version: Optional[str] = None,
        model_engine_map: Optional[Mapping[str, str]] = None,
        dest_url: Optional[str] = None,
    ):
        self.name = name if name else f"ep_{id(self)}"
        self.api_type: Optional[TYPE_API_TYPES] = api_type
        self.api_base = api_base
        self.api_key = api_key
        self.organization = organization
        self.api_version = api_version
        self.model_engine_map = model_engine_map
        self.dest_url = dest_url

    def __str__(self) -> str:
        # do not print api_key
        listed_attributes = [
            f"name={repr(self.name)}" if self.name else None,
            "api_key=*" if self.api_key else None,
            f"organization={repr(self.organization)}" if self.organization else None,
            f"api_base={repr(self.api_base)}" if self.api_base else None,
            f"api_type={repr(self.api_type)}" if self.api_type else None,
            f"api_version={repr(self.api_version)}" if self.api_version else None,
            f"model_engine_map={repr(self.model_engine_map)}"
            if self.model_engine_map
            else None,
            f"dest_url={repr(self.dest_url)}" if self.dest_url else None,
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

    def merge(self, other: Endpoint, override=False):
        if not isinstance(other, Endpoint):
            raise ValueError(f"Cannot merge with {type(other)}")
        if self.api_key is None or override:
            self.api_key = other.api_key
        if self.organization is None or override:
            self.organization = other.organization
        if self.api_base is None or override:
            self.api_base = other.api_base
        if self.api_type is None or override:
            self.api_type = other.api_type
        if self.api_version is None or override:
            self.api_version = other.api_version
        if self.model_engine_map is None or override:
            self.model_engine_map = other.model_engine_map
        if self.dest_url is None or override:
            self.dest_url = other.dest_url

    def merge_from_env(self):
        if self.api_key is None:
            self.api_key = os.environ.get("OPENAI_API_KEY")
        if self.organization is None:
            self.organization = os.environ.get("OPENAI_ORGANIZATION") or os.environ.get(
                "OPENAI_ORG_ID"
            )
        if self.api_base is None:
            self.api_base = os.environ.get("OPENAI_API_BASE")
        if self.api_type is None:
            self.api_type = cast(TYPE_API_TYPES, os.environ.get("OPENAI_API_TYPE"))
        if self.api_version is None:
            self.api_version = os.environ.get("OPENAI_API_VERSION")
        if self.model_engine_map is None:
            json_str = os.environ.get("MODEL_ENGINE_MAP")
            if json_str:
                try:
                    self.model_engine_map = json_loads(json_str)
                except JSONDecodeError:
                    pass


class EndpointManager(MutableSequence):
    def __init__(
        self, endpoints: Optional[Iterable] = None, load_path: Optional[PathType] = None
    ):
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
            obj = yaml_load(fin)
        if isinstance(obj, Mapping):
            obj = obj.get("endpoints", None)
        if obj:
            self.load_from_list(obj, override=override)
