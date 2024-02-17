import time

from .base_openai_api import BaseOpenAIAPI
from .api_request import api_request, poll
from . import utils

from . import _API_TYPES_AZURE


class OpenAIAPI(BaseOpenAIAPI):

    @classmethod
    def api_request_endpoint(
        cls,
        request_url, 
        **kwargs
        ):
        api_key, organization, api_base, api_type, api_version, engine, dest_url = cls.consume_kwargs(kwargs)
        url = utils.join_url(api_base, request_url)
        return api_request(url, api_key, organization=organization, api_type=api_type, dest_url=dest_url, **kwargs)
    
    @staticmethod
    def stream_chat_with_role(response):
        role = ''
        for data in response:
            try:
                message = data['choices'][0]['delta']
                if 'role' in message:
                    role = message['role']
                if 'content' in message:
                    text = message['content']
                    yield role, text
            except (KeyError, IndexError):
                pass
    
    @classmethod
    def stream_chat(cls, response):
        for _, text in cls.stream_chat_with_role(response):
            yield text
    
    @staticmethod
    def stream_completions(response):
        for data in response:
            try:
                yield data['choices'][0]['text']
            except (KeyError, IndexError):
                pass
    
    @classmethod
    def chat(cls, messages, logger=None, log_marks=[], **kwargs):
        api_key, organization, api_base, api_type, api_version, engine, dest_url = cls.consume_kwargs(kwargs)
        start_time = time.perf_counter()
        try:
            response = cls.api_request_endpoint(
                cls.get_chat_request_url(api_type, api_version, engine), 
                messages=messages, 
                method='post', 
                api_key=api_key,
                organization=organization,
                api_base=api_base,
                api_type=api_type,
                dest_url=dest_url,
                **kwargs
            )
            stream = kwargs.get('stream', False)
            response = cls._chat_log_response(logger, log_marks, kwargs, messages, start_time, response, stream)
            return response
        except Exception as e:
            cls._chat_log_exception(logger, log_marks, kwargs, messages, start_time, e)
            raise e

    
    @classmethod
    def completions(cls, prompt, logger=None, log_marks=[], **kwargs):
        api_key, organization, api_base, api_type, api_version, engine, dest_url = cls.consume_kwargs(kwargs)
        input_str = cls._completions_log_input(prompt, logger, log_marks, kwargs)
        start_time = time.time()
        try:
            response = cls.api_request_endpoint(
                cls.get_completions_request_url(api_type, api_version, engine), 
                prompt=prompt, 
                method='post', 
                api_key=api_key,
                organization=organization,
                api_base=api_base,
                api_type=api_type,
                dest_url=dest_url,
                **kwargs
            )
            stream = kwargs.get('stream', False)
            response = cls._completions_log_output(response, input_str, start_time, logger, stream)
            return response
        except Exception as e:
            cls._completions_log_exception(e, input_str, start_time, logger)
            raise e

    
    @classmethod
    def edits(cls, **kwargs):
        request_url = '/edits'
        return cls.api_request_endpoint(request_url, method='post', **kwargs)

    @classmethod
    def embeddings(cls, **kwargs):
        api_key, organization, api_base, api_type, api_version, engine, dest_url = cls.consume_kwargs(kwargs)
        request_url = cls.get_request_url('/embeddings', api_type, api_version, engine)
        return cls.api_request_endpoint(
            request_url, 
            method='post', 
            api_key=api_key,
            organization=organization,
            api_base=api_base,
            api_type=api_type,
            dest_url=dest_url,
            **kwargs
            )

    @classmethod
    def models_list(cls, **kwargs):
        api_key, organization, api_base, api_type, api_version, engine, dest_url = cls.consume_kwargs(kwargs)
        request_url = cls.get_request_url('/models', api_type, api_version, engine)
        return cls.api_request_endpoint(
            request_url, 
            method='get', 
            api_key=api_key,
            organization=organization,
            api_base=api_base,
            api_type=api_type,
            dest_url=dest_url,
            **kwargs
            )

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
        api_key, organization, api_base, api_type, api_version, engine, dest_url = cls.consume_kwargs(kwargs)
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
            dest_url=dest_url,
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

