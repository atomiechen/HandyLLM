import os
import requests

class OpenAIAPI:
    
    base_url = "https://api.openai.com/v1"
    api_key = os.environ.get('OPENAI_API_KEY')
    organization = None

    @staticmethod
    def api_request(url, api_key, organization=None, timeout=None, **kwargs):
        if api_key is None:
            raise Exception("OpenAI API key is not set")
        if url is None:
            raise Exception("OpenAI API url is not set")
        # 避免直接打印api_key
        plaintext_len = 8
        print(f"API request: url={url} api_key={api_key[:plaintext_len]}{'*'*(len(api_key)-plaintext_len)}")
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
            raise Exception(f"OpenAI API error ({response.status_code} {response.reason}): {message}")
        return response.json()

    @staticmethod
    def api_request_endpoint(request_url, endpoint_manager=None, **kwargs):
        if endpoint_manager != None:
            # 每次换服务器和key要同时换，保证服务器和key是对应的
            base_url, api_key, organization = endpoint_manager.get_endpoint()
        else:
            base_url = OpenAIAPI.base_url
            api_key = OpenAIAPI.api_key
            organization = OpenAIAPI.organization
        url = base_url + request_url
        return OpenAIAPI.api_request(url, api_key, organization=organization, **kwargs)
    
    @staticmethod
    def chat(timeout=None, endpoint_manager=None, **kwargs):
        request_url = '/chat/completions'
        return OpenAIAPI.api_request_endpoint(request_url, timeout=timeout, endpoint_manager=endpoint_manager, **kwargs)
    
    @staticmethod
    def completions(timeout=None, endpoint_manager=None, **kwargs):
        request_url = '/completions'
        return OpenAIAPI.api_request_endpoint(request_url, timeout=timeout, endpoint_manager=endpoint_manager, **kwargs)
    
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

