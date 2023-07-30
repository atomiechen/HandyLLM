import os
import time
from urllib.parse import quote_plus
import json
import copy

from .api_request import api_request, poll
from .endpoint_manager import Endpoint, EndpointManager
from .prompt_converter import PromptConverter
from . import utils

from . import _API_BASE_OPENAI, _API_TYPE_OPENAI, _API_TYPES_AZURE, _API_VERSION_AZURE


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
        return api_key or cls.api_key or os.environ.get('OPENAI_API_KEY')
    
    @classmethod
    def get_organization(cls, organization=None):
        return organization or cls.organization or os.environ.get('OPENAI_ORGANIZATION')
    
    @classmethod
    def get_api_base(cls, api_base=None):
        return api_base or cls.api_base or os.environ.get('OPENAI_API_BASE') or _API_BASE_OPENAI
    
    @classmethod
    def get_api_type_and_version(cls, api_type=None, api_version=None):
        api_type = api_type or cls.api_type or os.environ.get('OPENAI_API_TYPE') or _API_TYPE_OPENAI
        api_version = api_version or cls.api_version or os.environ.get('OPENAI_API_VERSION')
        if not api_version and api_type and api_type.lower() in _API_TYPES_AZURE:
            api_version = _API_VERSION_AZURE
        return api_type, api_version

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
        return api_request(url, api_key, organization=organization, api_type=api_type, **kwargs)
    
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
        api_key, organization, api_base, api_type, api_version, engine = cls.consume_kwargs(kwargs)
        if api_type and api_type.lower() in _API_TYPES_AZURE:
            request_url = f'/openai/images/generations:submit?api-version={api_version}'
            raw_response = True
        else:
            request_url = '/images/generations'
            raw_response = False
        response = cls.api_request_endpoint(
            request_url, 
            method='post', 
            api_key=api_key,
            organization=organization,
            api_base=api_base,
            api_type=api_type,
            raw_response=raw_response,
            **kwargs
            )
        if api_type and api_type.lower() in _API_TYPES_AZURE:
            # Azure image generation
            # use raw response to get poll_url
            poll_url = response.headers['operation-location']
            headers= { "api-key": api_key, "Content-Type": "application/json" }
            response = poll(
                url=poll_url, 
                method='get', 
                until=lambda response: response.json()['status'] == 'succeeded',
                failed=cls.check_image_failure,
                interval=lambda response: cls.get_retry(response) or 1,
                headers=headers, 
                ).json()
            return response.get('result', response)
        else:
            return response

    @staticmethod
    def check_image_failure(response):
        response_dict = response.json()
        if response_dict['status'] == 'failed':
            raise Exception(f"Image generation failed: {response_dict['error']['code']} {response_dict['error']['message']}")
    
    @staticmethod
    def get_retry(response):
        try:
            return int(response.headers.get('retry-after'))
        except:
            return None

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

