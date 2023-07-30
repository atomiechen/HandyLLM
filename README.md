# HandyLLM

[![GitHub](https://img.shields.io/badge/github-HandyLLM-blue?logo=github)](https://github.com/atomiechen/HandyLLM) [![PyPI](https://img.shields.io/pypi/v/HandyLLM?logo=pypi&logoColor=white)](https://pypi.org/project/HandyLLM/)

A handy toolkit for using LLM.



## Install

```shell
pip3 install handyllm
```

or, install from the Github repo to get latest updates:

```shell
pip3 install git+https://github.com/atomiechen/handyllm.git
```



## Examples

Example scripts are placed in [tests](./tests) folder.



## OpenAI API Request

### Endpoints

Each API request will connect to an endpoint along with some API configurations, which include: `api_key`, `organization`, `api_base`, `api_type` and `api_version`. 

An `Endpoint` object contains these information. An `EndpointManager` acts like a list and can be used to rotate the next endpoint. See [test_endpoint.py](./tests/test_endpoint.py).

There are 5 methods for specifying endpoint info:

1. (each API call) Pass these fields as keyword parameters.
2. (each API call) Pass an `endpoint` keyword parameter to specify an `Endpoint`.
3. (each API call) Pass an `endpoint_manager` keyword parameter to specify an `EndpointManager`.
4. (global) Set class variables: `OpenAIAPI.api_base`, `OpenAIAPI.api_key`, `OpenAIAPI.organization`, `OpenAIAPI.api_type`, `OpenAIAPI.api_version`.
5. (global) Set environment variables: `OPENAI_API_KEY`, `OPENAI_ORGANIZATION`, `OPENAI_API_BASE`, `OPENAI_API_TYPE`, `OPENAI_API_VERSION`.

**Note**: If a field is set to `None` in the previous method, it will be replaced by the non-`None` value in the subsequent method, until a default value is used (OpenAI's endpoint information).

**Azure OpenAI APIs are supported:** Specify `api_type='azure'`, and set `api_base` and `api_key` accordingly. See [test_azure.py](./tests/test_azure.py). Please refer to [Azure OpenAI Service Documentation](https://learn.microsoft.com/en-us/azure/ai-services/openai/) for details.

### Logger

You can pass custom `logger` and `log_marks` (a string or a collection of strings) to `chat`/`completions` to get input and output logging.

### Timeout control

This toolkit supports client-side `timeout` control:

```python
from handyllm import OpenAIAPI
prompt = [{
    "role": "user",
    "content": "please tell me a joke"
    }]
response = OpenAIAPI.chat(
    model="gpt-3.5-turbo",
    messages=prompt,
    timeout=10
    )
print(response['choices'][0]['message']['content'])
```

### Stream response

Stream response of `chat`/`completions`/`finetunes_list_events` can be achieved using `steam` parameter:

```python
response = OpenAIAPI.chat(
    model="gpt-3.5-turbo",
    messages=prompt,
    timeout=10,
    stream=True
    )

# you can use this to stream the response text
for text in OpenAIAPI.stream_chat(response):
    print(text, end='')

# or you can use this to get the whole response
# for chunk in response:
#     if 'content' in chunk['choices'][0]['delta']:
#         print(chunk['choices'][0]['delta']['content'], end='')
```

### Supported APIs

- chat
- completions
- edits
- embeddings
- models_list
- models_retrieve
- moderations
- images_generations
- images_edits
- images_variations
- audio_transcriptions
- audtio_translations
- files_list
- files_upload
- files_delete
- files_retrieve
- files_retrieve_content
- finetunes_create
- finetunes_list
- finetunes_retrieve
- finetunes_cancel
- finetunes_list_events
- finetunes_delete_model

Please refer to [OpenAI official API reference](https://platform.openai.com/docs/api-reference) for details.



## Prompt

### Prompt Conversion

`PromptConverter` can convert this text file `prompt.txt` into a structured prompt for chat API calls:

```
$system$
You are a helpful assistant.

$user$
Please help me merge the following two JSON documents into one.

$assistant$
Sure, please give me the two JSON documents.

$user$
{
    "item1": "It is really a good day."
}
{
    "item2": "Indeed."
}
%output_format%
%misc1%
%misc2%
```

```python
from handyllm import PromptConverter
converter = PromptConverter()

# chat can be used as the message parameter for OpenAI API
chat = converter.rawfile2chat('prompt.txt')

# variables wrapped in %s can be replaced at runtime
new_chat = converter.chat_replace_variables(
    chat, 
    {
        r'%misc1%': 'Note1: do not use any bad word.',
        r'%misc2%': 'Note2: be optimistic.',
    }
)
```

### Substitute

`PromptConverter` can also substitute placeholder variables like `%output_format%` stored in text files to make multiple prompts modular. A substitute map `substitute.txt` looks like this:

```
%output_format%
Please output a SINGLE JSON object that contains all items from the two input JSON objects.

%variable1%
Placeholder text.

%variable2%
Placeholder text.
```

```python
from handyllm import PromptConverter
converter = PromptConverter()
converter.read_substitute_content('substitute.txt')  # read substitute map
chat = converter.rawfile2chat('prompt.txt')  # variables are substituted already
```

