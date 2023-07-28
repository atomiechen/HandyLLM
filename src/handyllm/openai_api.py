import os
import time
import requests
import logging
import json
import copy

from .prompt_converter import PromptConverter

module_logger = logging.getLogger(__name__)

class OpenAIAPI:
    
    base_url = "https://api.openai.com/v1"
    
    # set this to your API key; 
    # or environment variable OPENAI_API_KEY will be used.
    api_key = None
    
    # set this to your organization ID; 
    # or environment variable OPENAI_ORGANIZATION will be used;
    # can be None.
    organization = None
    
    converter = PromptConverter()
    
    @staticmethod
    def _get_key_from_env():
        return os.environ.get('OPENAI_API_KEY')
    
    @staticmethod
    def _get_organization_from_env():
        return os.environ.get('OPENAI_ORGANIZATION')

    @staticmethod
    def _api_request(url, api_key, organization=None, method='post', timeout=None, **kwargs):
        if api_key is None:
            raise Exception("OpenAI API key is not set")
        if url is None:
            raise Exception("OpenAI API url is not set")

        ## log request info
        log_strs = []
        # 避免直接打印api_key
        plaintext_len = 8
        log_strs.append(f"API request {url}")
        log_strs.append(f"api_key: {api_key[:plaintext_len]}{'*'*(len(api_key)-plaintext_len)}")
        if organization is not None:
            log_strs.append(f"organization: {organization[:plaintext_len]}{'*'*(len(organization)-plaintext_len)}")
        log_strs.append(f"timeout: {timeout}")
        module_logger.info('\n'.join(log_strs))

        files = kwargs.pop('files', None)
        stream = kwargs.get('stream', False)
        headers = { 'Authorization': 'Bearer ' + api_key }
        json_data = None
        data = None
        params = None
        if organization is not None:
            headers['OpenAI-Organization'] = organization
        if method == 'post':
            if files is None:
                headers['Content-Type'] = 'application/json'
                json_data = kwargs
            else:  ## if files is not None, let requests handle the content type
                data = kwargs
        if method == 'get' and stream:
            params = { "stream": "true" }

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
    def stream_chat(response):
        for data in response:
            if 'content' in data['choices'][0]['delta']:
                yield data['choices'][0]['delta']['content']
    
    @staticmethod
    def stream_completions(response):
        for data in response:
            yield data['choices'][0]['text']
    
    @staticmethod
    def api_request_endpoint(request_url, endpoint_manager=None, **kwargs):
        specified_api_key = kwargs.pop('api_key', None)
        specified_organization = kwargs.pop('organization', None)
        if endpoint_manager != None:
            # 每次换服务器和key要同时换，保证服务器和key是对应的
            base_url, api_key, organization = endpoint_manager.get_endpoint()
        else:
            base_url = OpenAIAPI.base_url
            if specified_api_key is not None:
                api_key = specified_api_key
            elif OpenAIAPI.api_key is not None:
                api_key = OpenAIAPI.api_key
            else:
                api_key = OpenAIAPI._get_key_from_env()
            if specified_organization is not None:
                organization = specified_organization
            elif OpenAIAPI.organization is not None:
                organization = OpenAIAPI.organization 
            else:
                organization = OpenAIAPI._get_organization_from_env()
        url = base_url + request_url
        return OpenAIAPI._api_request(url, api_key, organization=organization, **kwargs)
    
    @staticmethod
    def chat(model, messages, timeout=None, endpoint_manager=None, logger=None, log_marks=[], **kwargs):
        request_url = '/chat/completions'

        if logger is not None and 'messages' in kwargs:
            arguments = copy.deepcopy(kwargs)
            arguments.pop('messages', None)
            input_lines = [str(item) for item in log_marks]
            input_lines.append(json.dumps(arguments, indent=2, ensure_ascii=False))
            input_lines.append(" INPUT START ".center(50, '-'))
            input_lines.append(OpenAIAPI.converter.chat2raw(kwargs['messages']))
            input_lines.append(" INPUT END ".center(50, '-')+"\n")
            input_str = "\n".join(input_lines)
        
        start_time = time.time()
        try:
            response = OpenAIAPI.api_request_endpoint(request_url, model=model, messages=messages, method='post', timeout=timeout, endpoint_manager=endpoint_manager, **kwargs)
            
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
                        for data in response:
                            if 'content' in data['choices'][0]['delta']:
                                text += data['choices'][0]['delta']['content']
                            yield data
                        log_strs.append(text)
                        log_strs.append(" OUTPUT END ".center(50, '-')+"\n")
                        logger.info('\n'.join(log_strs))
                    response = wrapper(response)
                else:
                    log_strs.append(response['choices'][0]['message']['content'])
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
    
    @staticmethod
    def completions(model, prompt, timeout=None, endpoint_manager=None, logger=None, log_marks=[], **kwargs):
        request_url = '/completions'

        if logger is not None and 'prompt' in kwargs:
            arguments = copy.deepcopy(kwargs)
            arguments.pop('prompt', None)
            input_lines = [str(item) for item in log_marks]
            input_lines.append(json.dumps(arguments, indent=2, ensure_ascii=False))
            input_lines.append(" INPUT START ".center(50, '-'))
            input_lines.append(kwargs['prompt'])
            input_lines.append(" INPUT END ".center(50, '-')+"\n")
            input_str = "\n".join(input_lines)
        
        start_time = time.time()
        try:
            response = OpenAIAPI.api_request_endpoint(request_url, model=model, prompt=prompt, method='post', timeout=timeout, endpoint_manager=endpoint_manager, **kwargs)

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
    
    @staticmethod
    def edits(model, instruction, timeout=None, endpoint_manager=None, **kwargs):
        request_url = '/edits'
        return OpenAIAPI.api_request_endpoint(request_url, model=model, instruction=instruction, method='post', timeout=timeout, endpoint_manager=endpoint_manager, **kwargs)

    @staticmethod
    def embeddings(model, input, timeout=None, endpoint_manager=None, **kwargs):
        request_url = '/embeddings'
        return OpenAIAPI.api_request_endpoint(request_url, model=model, input=input, method='post', timeout=timeout, endpoint_manager=endpoint_manager, **kwargs)

    @staticmethod
    def models_list(timeout=None, endpoint_manager=None, **kwargs):
        request_url = '/models'
        return OpenAIAPI.api_request_endpoint(request_url, method='get', timeout=timeout, endpoint_manager=endpoint_manager, **kwargs)

    @staticmethod
    def models_retrieve(model, timeout=None, endpoint_manager=None, **kwargs):
        request_url = f'/models/{model}'
        return OpenAIAPI.api_request_endpoint(request_url, method='get', timeout=timeout, endpoint_manager=endpoint_manager, **kwargs)

    @staticmethod
    def moderations(input, timeout=None, endpoint_manager=None, **kwargs):
        request_url = '/moderations'
        return OpenAIAPI.api_request_endpoint(request_url, input=input, method='post', timeout=timeout, endpoint_manager=endpoint_manager, **kwargs)

    @staticmethod
    def images_generations(prompt, timeout=None, endpoint_manager=None, **kwargs):
        request_url = '/images/generations'
        return OpenAIAPI.api_request_endpoint(request_url, prompt=prompt, method='post', timeout=timeout, endpoint_manager=endpoint_manager, **kwargs)

    @staticmethod
    def images_edits(image, prompt, mask=None, timeout=None, endpoint_manager=None, **kwargs):
        request_url = '/images/edits'
        files = { 'image': image }
        if mask:
            files['mask'] = mask
        return OpenAIAPI.api_request_endpoint(request_url, prompt=prompt, method='post', files=files, timeout=timeout, endpoint_manager=endpoint_manager, **kwargs)

    @staticmethod
    def images_variations(image, timeout=None, endpoint_manager=None, **kwargs):
        request_url = '/images/variations'
        files = { 'image': image }
        return OpenAIAPI.api_request_endpoint(request_url, method='post', files=files, timeout=timeout, endpoint_manager=endpoint_manager, **kwargs)

    @staticmethod
    def audio_transcriptions(file, model, timeout=None, endpoint_manager=None, **kwargs):
        request_url = '/audio/transcriptions'
        files = { 'file': file }
        return OpenAIAPI.api_request_endpoint(request_url, model=model, method='post', files=files, timeout=timeout, endpoint_manager=endpoint_manager, **kwargs)

    @staticmethod
    def audio_translations(file, model, timeout=None, endpoint_manager=None, **kwargs):
        request_url = '/audio/translations'
        files = { 'file': file }
        return OpenAIAPI.api_request_endpoint(request_url, model=model, method='post', files=files, timeout=timeout, endpoint_manager=endpoint_manager, **kwargs)

    @staticmethod
    def files_list(timeout=None, endpoint_manager=None, **kwargs):
        request_url = '/files'
        return OpenAIAPI.api_request_endpoint(request_url, method='get', timeout=timeout, endpoint_manager=endpoint_manager, **kwargs)

    @staticmethod
    def files_upload(file, purpose, timeout=None, endpoint_manager=None, **kwargs):
        request_url = '/files'
        files = { 'file': file }
        return OpenAIAPI.api_request_endpoint(request_url, purpose=purpose, method='post', files=files, timeout=timeout, endpoint_manager=endpoint_manager, **kwargs)

    @staticmethod
    def files_delete(file_id, timeout=None, endpoint_manager=None, **kwargs):
        request_url = f'/files/{file_id}'
        return OpenAIAPI.api_request_endpoint(request_url, method='delete', timeout=timeout, endpoint_manager=endpoint_manager, **kwargs)

    @staticmethod
    def files_retrieve(file_id, timeout=None, endpoint_manager=None, **kwargs):
        request_url = f'/files/{file_id}'
        return OpenAIAPI.api_request_endpoint(request_url, method='get', timeout=timeout, endpoint_manager=endpoint_manager, **kwargs)

    @staticmethod
    def files_retrieve_content(file_id, timeout=None, endpoint_manager=None, **kwargs):
        request_url = f'/files/{file_id}/content'
        return OpenAIAPI.api_request_endpoint(request_url, method='get', timeout=timeout, endpoint_manager=endpoint_manager, **kwargs)

    @staticmethod
    def finetunes_create(training_file, timeout=None, endpoint_manager=None, **kwargs):
        request_url = '/fine-tunes'
        return OpenAIAPI.api_request_endpoint(request_url, training_file=training_file, method='post', timeout=timeout, endpoint_manager=endpoint_manager, **kwargs)

    @staticmethod
    def finetunes_list(timeout=None, endpoint_manager=None, **kwargs):
        request_url = '/fine-tunes'
        return OpenAIAPI.api_request_endpoint(request_url, method='get', timeout=timeout, endpoint_manager=endpoint_manager, **kwargs)

    @staticmethod
    def finetunes_retrieve(fine_tune_id, timeout=None, endpoint_manager=None, **kwargs):
        request_url = f'/fine-tunes/{fine_tune_id}'
        return OpenAIAPI.api_request_endpoint(request_url, method='get', timeout=timeout, endpoint_manager=endpoint_manager, **kwargs)

    @staticmethod
    def finetunes_cancel(fine_tune_id, timeout=None, endpoint_manager=None, **kwargs):
        request_url = f'/fine-tunes/{fine_tune_id}/cancel'
        return OpenAIAPI.api_request_endpoint(request_url, method='post', timeout=timeout, endpoint_manager=endpoint_manager, **kwargs)

    @staticmethod
    def finetunes_list_events(fine_tune_id, timeout=None, endpoint_manager=None, **kwargs):
        request_url = f'/fine-tunes/{fine_tune_id}/events'
        return OpenAIAPI.api_request_endpoint(request_url, method='get', timeout=timeout, endpoint_manager=endpoint_manager, **kwargs)

    @staticmethod
    def finetunes_delete_model(model, timeout=None, endpoint_manager=None, **kwargs):
        request_url = f'/models/{model}'
        return OpenAIAPI.api_request_endpoint(request_url, method='delete', timeout=timeout, endpoint_manager=endpoint_manager, **kwargs)


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

