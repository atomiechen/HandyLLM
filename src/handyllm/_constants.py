from typing import Literal

API_BASE_OPENAI = 'https://api.openai.com/v1'
API_TYPE_OPENAI = 'openai'
API_TYPES_AZURE = (
    'azure', 
    'azure_ad', 
    'azuread'
)

TYPE_API_TYPES = Literal['openai', 'azure', 'azure_ad', 'azuread']
