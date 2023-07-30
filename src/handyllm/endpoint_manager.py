from threading import Lock
from collections.abc import MutableSequence


class Endpoint:
    def __init__(
        self, 
        name=None,
        api_key=None, 
        organization=None, 
        api_base=None, 
        api_type=None,
        api_version=None, 
        ):
        self.name = name if name else f"ep_{id(self)}"
        self.api_key = api_key
        self.organization = organization
        self.api_base = api_base
        self.api_type = api_type
        self.api_version = api_version

    def __str__(self) -> str:
        # do not print api_key
        listed_attributes = [
            f'name={repr(self.name)}' if self.name else None,
            f'api_key=*' if self.api_key else None,
            f'organization={repr(self.organization)}' if self.organization else None,
            f'api_base={repr(self.api_base)}' if self.api_base else None,
            f'api_type={repr(self.api_type)}' if self.api_type else None,
            f'api_version={repr(self.api_version)}' if self.api_version else None,
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
            self.api_version
        )


class EndpointManager(MutableSequence):

    def __init__(self):
        self._lock = Lock()
        self._last_idx_endpoint = 0
        self._endpoints = []

    def clear(self):
        self._last_idx_endpoint = 0
        self._endpoints.clear()
        
    def __len__(self) -> int:
        return len(self._endpoints)

    def __getitem__(self, idx: int) -> Endpoint:
        return self._endpoints[idx]

    def __setitem__(self, idx: int, endpoint: Endpoint):
        self._endpoints[idx] = endpoint

    def __delitem__(self, idx: int):
        del self._endpoints[idx]

    def insert(self, idx: int, endpoint: Endpoint):
        self._endpoints.insert(idx, endpoint)

    def add_endpoint_by_info(self, **kwargs):
        endpoint = Endpoint(**kwargs)
        self.append(endpoint)

    def get_next_endpoint(self) -> Endpoint:
        with self._lock:
            endpoint = self._endpoints[self._last_idx_endpoint]
            if self._last_idx_endpoint == len(self._endpoints) - 1:
                self._last_idx_endpoint = 0
            else:
                self._last_idx_endpoint += 1
            return endpoint

