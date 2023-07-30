import os
import time
from urllib.parse import quote_plus
import requests
import logging
import json
import copy

from .endpoint_manager import Endpoint, EndpointManager
from .prompt_converter import PromptConverter
from . import utils

_API_BASE_OPENAI = 'https://api.openai.com/v1'
_API_TYPE_OPENAI = 'openai'
_API_TYPES_AZURE = (
    'azure', 
    'azure_ad', 
    'azuread'
)
_API_VERSION_AZURE = '2023-05-15'

module_logger = logging.getLogger(__name__)

class OpenAIAPI:
    
    # deprecated
    base_url = _API_BASE_OPENAI
    
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
    
    converter = PromptConverter()
    
    @classmethod
    def get_api_key(cls, api_key=None):
        if not api_key:
            api_key = cls.api_key if cls.api_key else os.environ.get('OPENAI_API_KEY')
        return api_key
    
    @classmethod
    def get_organization(cls, organization=None):
        if not organization:
            organization = cls.organization if cls.organization else os.environ.get('OPENAI_ORGANIZATION')
        return organization
    
    @classmethod
    def get_api_base(cls, api_base=None):
        if not api_base:
            api_base = cls.api_base if cls.api_base else os.environ.get('OPENAI_API_BASE')
            if not api_base:
                api_base = _API_BASE_OPENAI
        return api_base
    
    @classmethod
    def get_api_type_and_version(cls, api_type=None, api_version=None):
        if not api_type:
            api_type = cls.api_type if cls.api_type else os.environ.get('OPENAI_API_TYPE')
            if not api_type:
                api_type = _API_TYPE_OPENAI
        if not api_version:
            api_version = cls.api_version if cls.api_version else os.environ.get('OPENAI_API_VERSION')
            if not api_version and api_type and api_type.lower() in _API_TYPES_AZURE:
                api_version = _API_VERSION_AZURE
        return api_type, api_version

    @staticmethod
    def _api_request(url, api_key, organization=None, api_type=api_type, method='post', timeout=None, **kwargs):
        if api_key is None:
            raise Exception("OpenAI API key is not set")
        if url is None:
            raise Exception("OpenAI API url is not set")
        if api_type is None:
            raise Exception("OpenAI API type is not set")

        ## log request info
        log_strs = []
        # avoid logging the whole api_key
        plaintext_len = 8
        log_strs.append(f"API request {url}")
        log_strs.append(f"api_key: {api_key[:plaintext_len]}{'*'*(len(api_key)-plaintext_len)}")
        if organization is not None:
            log_strs.append(f"organization: {organization[:plaintext_len]}{'*'*(len(organization)-plaintext_len)}")
        log_strs.append(f"timeout: {timeout}")
        module_logger.info('\n'.join(log_strs))

        files = kwargs.pop('files', None)
        stream = kwargs.get('stream', False)
        headers = {}
        json_data = None
        data = None
        params = {}
        if api_type in _API_TYPES_AZURE:
            headers['api-key'] = api_key
        else:
            headers['Authorization'] = 'Bearer ' + api_key
        if organization is not None:
            headers['OpenAI-Organization'] = organization
        if method == 'post':
            if files is None:
                headers['Content-Type'] = 'application/json'
                json_data = kwargs
            else:  ## if files is not None, let requests handle the content type
                data = kwargs
        if method == 'get' and stream:
            params['stream'] = 'true'

        response = requests.request(
            method,
            url,
            headers=headers,
            data=data,
            json=json_data,
            files=files,
            params=params,
            stream=stream,
            timeout=timeout,
            )
        if response.status_code != 200:
            # report both status code and error message
            try:
                message = response.json()['error']['message']
            except:
                message = response.text
            err_msg = f"OpenAI API error ({url} {response.status_code} {response.reason}): {message}"
            module_logger.error(err_msg)
            raise Exception(err_msg)

        if stream:
            return OpenAIAPI._gen_stream_response(response)
        else:
            return response.json()

    @staticmethod
    def _gen_stream_response(response):
        for byte_line in response.iter_lines():  # do not auto decode
            if byte_line:
                if byte_line.strip() == b"data: [DONE]":
                    return
                if byte_line.startswith(b"data: "):
                    line = byte_line[len(b"data: "):].decode("utf-8")
                    yield json.loads(line)

    @staticmethod
    def stream_chat_with_role(response):
        role = ''
        for data in response:
            message = data['choices'][0]['delta']
            if 'role' in message:
                role = message['role']
            if 'content' in message:
                text = message['content']
                yield role, text
    
    @staticmethod
    def stream_chat(response):
        for _, text in OpenAIAPI.stream_chat_with_role(response):
            yield text
    
    @staticmethod
    def stream_completions(response):
        for data in response:
            yield data['choices'][0]['text']
    
    @classmethod
    def api_request_endpoint(
        cls,
        request_url, 
        **kwargs
        ):
        api_key, organization, api_base, api_type, api_version, engine = cls.consume_kwargs(kwargs)
        url = utils.join_url(api_base, request_url)
        return cls._api_request(url, api_key, organization=organization, api_type=api_type, **kwargs)
    
    @classmethod
    def consume_kwargs(cls, kwargs):
        api_key = organization = api_base = api_type = api_version = engine = None

        # read API info from endpoint_manager
        endpoint_manager = kwargs.pop('endpoint_manager', None)
        if endpoint_manager is not None:
            if not isinstance(endpoint_manager, EndpointManager):
                raise Exception("endpoint_manager must be an instance of EndpointManager")
            # get_next_endpoint() will be called once for each request
            api_key, organization, api_base, api_type, api_version = endpoint_manager.get_next_endpoint().get_api_info()

        # read API info from endpoint (override API info from endpoint_manager)
        endpoint = kwargs.pop('endpoint', None)
        if endpoint is not None:
            if not isinstance(endpoint, Endpoint):
                raise Exception("endpoint must be an instance of Endpoint")
            api_key, organization, api_base, api_type, api_version = endpoint.get_api_info()

        # read API info from kwargs, class variables, and environment variables
        api_key = cls.get_api_key(kwargs.pop('api_key', api_key))
        organization = cls.get_organization(kwargs.pop('organization', organization))
        api_base = cls.get_api_base(kwargs.pop('api_base', api_base))
        api_type, api_version = cls.get_api_type_and_version(
            kwargs.pop('api_type', api_type), 
            kwargs.pop('api_version', api_version)
        )

        deployment_id = kwargs.pop('deployment_id', None)
        engine = kwargs.pop('engine', deployment_id)
        return api_key, organization, api_base, api_type, api_version, engine
    
    @classmethod
    def chat(cls, messages, logger=None, log_marks=[], **kwargs):
        api_key, organization, api_base, api_type, api_version, engine = cls.consume_kwargs(kwargs)
        if api_type and api_type.lower() in _API_TYPES_AZURE:
            if engine is None:
                raise Exception("Azure API requires engine to be specified")
            request_url = f'/openai/deployments/{quote_plus(engine)}/chat/completions?api-version={api_version}'
        else:
            if engine is not None:
                request_url = f'/engines/{quote_plus(engine)}/chat/completions'
            else:
                request_url = '/chat/completions'

        if logger is not None:
            arguments = copy.deepcopy(kwargs)
            # check if log_marks is iterable
            if utils.isiterable(log_marks):
                input_lines = [str(item) for item in log_marks]
            else:
                input_lines = [str(log_marks)]
            input_lines.append(json.dumps(arguments, indent=2, ensure_ascii=False))
            input_lines.append(" INPUT START ".center(50, '-'))
            input_lines.append(cls.converter.chat2raw(messages))
            input_lines.append(" INPUT END ".center(50, '-')+"\n")
            input_str = "\n".join(input_lines)
        
        start_time = time.time()
        try:
            response = cls.api_request_endpoint(
                request_url, 
                messages=messages, 
                method='post', 
                api_key=api_key,
                organization=organization,
                api_base=api_base,
                api_type=api_type,
                **kwargs
            )
            
            if logger is not None:
                end_time = time.time()
                ## log this on result
                log_strs = []
                log_strs.append(f"Chat request result ({end_time-start_time:.2f}s)")
                log_strs.append(input_str)

                log_strs.append(" OUTPUT START ".center(50, '-'))
                stream = kwargs.get('stream', False)
                if stream:
                    def wrapper(response):
                        text = ''
                        role = ''
                        for data in response:
                            message = data['choices'][0]['delta']
                            if 'role' in message:
                                role = message['role']
                            if 'content' in message:
                                text += message['content']
                            yield data
                        log_strs.append(cls.converter.chat2raw([{'role': role, 'content': text}]))
                        log_strs.append(" OUTPUT END ".center(50, '-')+"\n")
                        logger.info('\n'.join(log_strs))
                    response = wrapper(response)
                else:
                    log_strs.append(cls.converter.chat2raw([response['choices'][0]['message']]))
                    log_strs.append(" OUTPUT END ".center(50, '-')+"\n")
                    logger.info('\n'.join(log_strs))
        except Exception as e:
            if logger is not None:
                end_time = time.time()
                log_strs = []
                log_strs.append(f"Chat request error ({end_time-start_time:.2f}s)")
                log_strs.append(input_str)
                log_strs.append(str(e))
                logger.error('\n'.join(log_strs))
            raise e

        return response
    
    @classmethod
    def completions(cls, prompt, logger=None, log_marks=[], **kwargs):
        api_key, organization, api_base, api_type, api_version, engine = cls.consume_kwargs(kwargs)
        if api_type and api_type.lower() in _API_TYPES_AZURE:
            if engine is None:
                raise Exception("Azure API requires engine to be specified")
            request_url = f'/openai/deployments/{quote_plus(engine)}/completions?api-version={api_version}'
        else:
            if engine is not None:
                request_url = f'/engines/{quote_plus(engine)}/completions'
            else:
                request_url = '/completions'

        if logger is not None:
            arguments = copy.deepcopy(kwargs)
            # check if log_marks is iterable
            if utils.isiterable(log_marks):
                input_lines = [str(item) for item in log_marks]
            else:
                input_lines = [str(log_marks)]
            input_lines.append(json.dumps(arguments, indent=2, ensure_ascii=False))
            input_lines.append(" INPUT START ".center(50, '-'))
            input_lines.append(prompt)
            input_lines.append(" INPUT END ".center(50, '-')+"\n")
            input_str = "\n".join(input_lines)
        
        start_time = time.time()
        try:
            response = cls.api_request_endpoint(
                request_url, 
                prompt=prompt, 
                method='post', 
                api_key=api_key,
                organization=organization,
                api_base=api_base,
                api_type=api_type,
                **kwargs
            )

            if logger is not None:
                end_time = time.time()
                ## log this on result
                log_strs = []
                log_strs.append(f"Completions request result ({end_time-start_time:.2f}s)")
                log_strs.append(input_str)

                log_strs.append(" OUTPUT START ".center(50, '-'))
                stream = kwargs.get('stream', False)
                if stream:
                    def wrapper(response):
                        text = ''
                        for data in response:
                            text += data['choices'][0]['text']
                            yield data
                        log_strs.append(text)
                        log_strs.append(" OUTPUT END ".center(50, '-')+"\n")
                        logger.info('\n'.join(log_strs))
                    response = wrapper(response)
                else:
                    log_strs.append(response['choices'][0]['text'])
                    log_strs.append(" OUTPUT END ".center(50, '-')+"\n")
                    logger.info('\n'.join(log_strs))
        except Exception as e:
            if logger is not None:
                end_time = time.time()
                log_strs = []
                log_strs.append(f"Completions request error ({end_time-start_time:.2f}s)")
                log_strs.append(input_str)
                log_strs.append(str(e))
                logger.error('\n'.join(log_strs))
            raise e

        return response
    
    @classmethod
    def edits(cls, **kwargs):
        request_url = '/edits'
        return cls.api_request_endpoint(request_url, method='post', **kwargs)

    @classmethod
    def embeddings(cls, **kwargs):
        api_key, organization, api_base, api_type, api_version, engine = cls.consume_kwargs(kwargs)
        if api_type and api_type.lower() in _API_TYPES_AZURE:
            if engine is None:
                raise Exception("Azure API requires engine to be specified")
            request_url = f'/openai/deployments/{quote_plus(engine)}/embeddings?api-version={api_version}'
        else:
            if engine is not None:
                request_url = f'/engines/{quote_plus(engine)}/embeddings'
            else:
                request_url = '/embeddings'
        return cls.api_request_endpoint(
            request_url, 
            method='post', 
            api_key=api_key,
            organization=organization,
            api_base=api_base,
            api_type=api_type,
            **kwargs
            )

    @classmethod
    def models_list(cls, **kwargs):
        request_url = '/models'
        return cls.api_request_endpoint(request_url, method='get', **kwargs)

    @classmethod
    def models_retrieve(cls, model, **kwargs):
        request_url = f'/models/{model}'
        return cls.api_request_endpoint(request_url, method='get', **kwargs)

    @classmethod
    def moderations(cls, **kwargs):
        request_url = '/moderations'
        return cls.api_request_endpoint(request_url, method='post', **kwargs)

    @classmethod
    def images_generations(cls, **kwargs):
        request_url = '/images/generations'
        return cls.api_request_endpoint(request_url, method='post', **kwargs)

    @classmethod
    def images_edits(cls, image, mask=None, **kwargs):
        request_url = '/images/edits'
        files = { 'image': image }
        if mask:
            files['mask'] = mask
        return cls.api_request_endpoint(request_url, method='post', files=files, **kwargs)

    @classmethod
    def images_variations(cls, image, **kwargs):
        request_url = '/images/variations'
        files = { 'image': image }
        return cls.api_request_endpoint(request_url, method='post', files=files, **kwargs)

    @classmethod
    def audio_transcriptions(cls, file, **kwargs):
        request_url = '/audio/transcriptions'
        files = { 'file': file }
        return cls.api_request_endpoint(request_url, method='post', files=files, **kwargs)

    @classmethod
    def audio_translations(cls, file, **kwargs):
        request_url = '/audio/translations'
        files = { 'file': file }
        return cls.api_request_endpoint(request_url, method='post', files=files, **kwargs)

    @classmethod
    def files_list(cls, **kwargs):
        request_url = '/files'
        return cls.api_request_endpoint(request_url, method='get', **kwargs)

    @classmethod
    def files_upload(cls, file, **kwargs):
        request_url = '/files'
        files = { 'file': file }
        return cls.api_request_endpoint(request_url, method='post', files=files, **kwargs)

    @classmethod
    def files_delete(cls, file_id, **kwargs):
        request_url = f'/files/{file_id}'
        return cls.api_request_endpoint(request_url, method='delete', **kwargs)

    @classmethod
    def files_retrieve(cls, file_id, **kwargs):
        request_url = f'/files/{file_id}'
        return cls.api_request_endpoint(request_url, method='get', **kwargs)

    @classmethod
    def files_retrieve_content(cls, file_id, **kwargs):
        request_url = f'/files/{file_id}/content'
        return cls.api_request_endpoint(request_url, method='get', **kwargs)

    @classmethod
    def finetunes_create(cls, **kwargs):
        request_url = '/fine-tunes'
        return cls.api_request_endpoint(request_url, method='post', **kwargs)

    @classmethod
    def finetunes_list(cls, **kwargs):
        request_url = '/fine-tunes'
        return cls.api_request_endpoint(request_url, method='get', **kwargs)

    @classmethod
    def finetunes_retrieve(cls, fine_tune_id, **kwargs):
        request_url = f'/fine-tunes/{fine_tune_id}'
        return cls.api_request_endpoint(request_url, method='get', **kwargs)

    @classmethod
    def finetunes_cancel(cls, fine_tune_id, **kwargs):
        request_url = f'/fine-tunes/{fine_tune_id}/cancel'
        return cls.api_request_endpoint(request_url, method='post', **kwargs)

    @classmethod
    def finetunes_list_events(cls, fine_tune_id, **kwargs):
        request_url = f'/fine-tunes/{fine_tune_id}/events'
        return cls.api_request_endpoint(request_url, method='get', **kwargs)

    @classmethod
    def finetunes_delete_model(cls, model, **kwargs):
        request_url = f'/models/{model}'
        return cls.api_request_endpoint(request_url, method='delete', **kwargs)


if __name__ == '__main__':
    # OpenAIAPI.api_key = 'sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
    prompt = [{
        "role": "user",
        "content": "please tell me a joke"
        }]
    response = OpenAIAPI.chat(
        model="gpt-3.5-turbo-0301",
        messages=prompt,
        temperature=0.2,
        max_tokens=256,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        timeout=10
        )
    print(response)
    print(response['choices'][0]['message']['content'])
    
    ## below for comparison
    # import openai
    # response = openai.ChatCompletion.create(
    #     model="gpt-3.5-turbo-0301",
    #     messages=prompt,
    #     temperature=1.2,
    #     max_tokens=256,
    #     top_p=1.0,
    #     frequency_penalty=0.0,
    #     presence_penalty=0.0,
    #     api_key=openai_api_key,
    #     timeout=10  ## this is not working
    # )
    # print(response)
    # print(response['choices'][0]['message']['content'])

