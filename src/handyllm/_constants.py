from typing import Literal

_API_BASE_OPENAI = 'https://api.openai.com/v1'
_API_TYPE_OPENAI = 'openai'
_API_TYPES_AZURE = (
    'azure', 
    'azure_ad', 
    'azuread'
)

TYPE_API_TYPES = Literal['openai', 'azure', 'azure_ad', 'azuread']
