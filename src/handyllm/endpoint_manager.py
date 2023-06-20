from threading import Lock
from . import OpenAIAPI

class EndpointManager:
    
    def __init__(self):
        self._lock = Lock()
        
        self._base_urls = []
        self._last_idx_url = 0
        
        self._keys = []
        self._last_idx_key = 0
        
        self._organizations = []
        self._last_idx_organization = 0

    def add_base_url(self, base_url: str):
        if isinstance(base_url, str) and base_url.strip() != '':
            self._base_urls.append(base_url)

    def add_key(self, key: str):
        if isinstance(key, str) and key.strip() != '':
            self._keys.append(key)
    
    def add_organization(self, organization: str):
        if isinstance(organization, str) and organization.strip() != '':
            self._organizations.append(organization)
    
    def set_base_urls(self, base_urls):
        self._base_urls = [url for url in base_urls if isinstance(url, str) and url.strip() != '']
    
    def set_keys(self, keys):
        self._keys = [key for key in keys if isinstance(key, str) and key.strip() != '']
        
    def set_organizations(self, organizations):
        self._organizations = [organization for organization in organizations if isinstance(organization, str) and organization.strip() != '']

    def get_base_url(self):
        if len(self._base_urls) == 0:
            return OpenAIAPI.base_url
        else:
            base_url = self._base_urls[self._last_idx_url]
            if self._last_idx_url == len(self._base_urls) - 1:
                self._last_idx_url = 0
            else:
                self._last_idx_url += 1
            return base_url

    def get_key(self):
        if len(self._keys) == 0:
            return OpenAIAPI.api_key
        else:
            key = self._keys[self._last_idx_key]
            if self._last_idx_key == len(self._keys) - 1:
                self._last_idx_key = 0
            else:
                self._last_idx_key += 1
            return key
    
    def get_organization(self):
        if len(self._organizations) == 0:
            return OpenAIAPI.organization
        else:
            organization = self._organizations[self._last_idx_organization]
            if self._last_idx_organization == len(self._keys) - 1:
                self._last_idx_organization = 0
            else:
                self._last_idx_organization += 1
            return organization
    
    def get_endpoint(self):
        with self._lock:
            # compose full url
            base_url = self.get_base_url()
            # get API key
            api_key = self.get_key()
            # get organization
            organization = self.get_organization()
            return base_url, api_key, organization

