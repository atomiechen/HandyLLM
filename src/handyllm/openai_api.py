import time
import json
import copy

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
        request_url = cls.get_request_url('/chat/completions', api_type, api_version, engine)

        input_str = cls.chat_log_prepare(messages, logger, log_marks, kwargs)
        
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
                dest_url=dest_url,
                **kwargs
            )
            
            stream = kwargs.get('stream', False)
            cls.chat_log_response(response, input_str, start_time, logger, stream)
        except Exception as e:
            cls.chat_log_exception(e, input_str, start_time, logger)
            raise e

        return response
    
    @classmethod
    def completions(cls, prompt, logger=None, log_marks=[], **kwargs):
        api_key, organization, api_base, api_type, api_version, engine, dest_url = cls.consume_kwargs(kwargs)
        request_url = cls.get_request_url('/completions', api_type, api_version, engine)

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
                dest_url=dest_url,
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
                            try:
                                text += data['choices'][0]['text']
                            except (KeyError, IndexError):
                                pass
                            yield data
                        log_strs.append(text)
                        log_strs.append(" OUTPUT END ".center(50, '-')+"\n")
                        logger.info('\n'.join(log_strs))
                    response = wrapper(response)
                else:
                    try:
                        log_strs.append(response['choices'][0]['text'])
                    except (KeyError, IndexError):
                        log_strs.append("Wrong response format, no text found")
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

