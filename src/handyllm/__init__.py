_API_BASE_OPENAI = 'https://api.openai.com/v1'
_API_TYPE_OPENAI = 'openai'
_API_TYPES_AZURE = (
    'azure', 
    'azure_ad', 
    'azuread'
)
_API_VERSION_AZURE = '2023-05-15'

from .openai_api import OpenAIAPI
from .endpoint_manager import Endpoint, EndpointManager
from .prompt_converter import PromptConverter
