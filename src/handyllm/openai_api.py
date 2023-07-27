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
    def _api_request(url, api_key, organization=None, timeout=None, **kwargs):
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

        request_data = kwargs
        headers = {
            'Authorization': 'Bearer ' + api_key,
            'Content-Type': 'application/json'
            }
        if organization is not None:
            headers['OpenAI-Organization'] = organization
        
        response = requests.post(
            url, 
            headers=headers, 
            # data=json.dumps(request_data),
            json=request_data,
            timeout=timeout
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

        stream = kwargs.get('stream', False)
        if stream:
            return OpenAIAPI._gen_stream_response(response)
        else:
            return response.json()

    @staticmethod
    def _gen_stream_response(response):
        data_buffer = ''
        for chunk in response.iter_content(decode_unicode=True):
            data_buffer += chunk
            while '\n' in data_buffer:  # when '\n' is in the buffer, there is a complete message to process
                line, data_buffer = data_buffer.split('\n', 1)
                line = line.strip()
                if line.startswith('data:'):
                    line = line[len('data:'):].strip()
                    if line == '[DONE]':  # end the function when '[DONE]' message is received
                        return
                    else:
                        data = json.loads(line)
                        yield data

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
        if endpoint_manager != None:
            # 每次换服务器和key要同时换，保证服务器和key是对应的
            base_url, api_key, organization = endpoint_manager.get_endpoint()
        else:
            base_url = OpenAIAPI.base_url
            api_key = OpenAIAPI.api_key if OpenAIAPI.api_key is not None else OpenAIAPI._get_key_from_env()
            organization = OpenAIAPI.organization if OpenAIAPI.organization is not None else OpenAIAPI._get_organization_from_env()
        url = base_url + request_url
        return OpenAIAPI._api_request(url, api_key, organization=organization, **kwargs)
    
    @staticmethod
    def chat(timeout=None, endpoint_manager=None, logger=None, log_marks=[], **kwargs):
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
            response = OpenAIAPI.api_request_endpoint(request_url, timeout=timeout, endpoint_manager=endpoint_manager, **kwargs)
            
            if logger is not None:
                end_time = time.time()
                ## log this on result
                log_strs = []
                log_strs.append(f"Chat request result ({end_time-start_time:.2f}s)")
                log_strs.append(input_str)

                log_strs.append(" OUTPUT START ".center(50, '-'))
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
    def completions(timeout=None, endpoint_manager=None, logger=None, log_marks=[], **kwargs):
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
            response = OpenAIAPI.api_request_endpoint(request_url, timeout=timeout, endpoint_manager=endpoint_manager, **kwargs)

            if logger is not None:
                end_time = time.time()
                ## log this on result
                log_strs = []
                log_strs.append(f"Completions request result ({end_time-start_time:.2f}s)")
                log_strs.append(input_str)

                log_strs.append(" OUTPUT START ".center(50, '-'))
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
    def embeddings(timeout=None, endpoint_manager=None, **kwargs):
        request_url = '/embeddings'
        return OpenAIAPI.api_request_endpoint(request_url, timeout=timeout, endpoint_manager=endpoint_manager, **kwargs)


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

