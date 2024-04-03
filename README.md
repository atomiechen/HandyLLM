# HandyLLM

[![GitHub](https://img.shields.io/badge/github-HandyLLM-blue?logo=github)](https://github.com/atomiechen/HandyLLM) [![PyPI](https://img.shields.io/pypi/v/HandyLLM?logo=pypi&logoColor=white)](https://pypi.org/project/HandyLLM/)

A handy toolkit for using LLM.



## 🌟 Why HandyLLM?

☯️ Both sync and async APIs supported with unified design

🍡 OpenAI and Azure APIs all in one

☕️ Easy life with API endpoint management

📃 Writing chat prompt in a human-friendly mark-up format



## Installation

```shell
pip3 install handyllm
```

or, install from the Github repo to get latest updates:

```shell
pip3 install git+https://github.com/atomiechen/handyllm.git
```



## Usage

More example scripts are placed in [tests](./tests) folder.

### Using OpenAIClient

Each API method of `OpenAIClient` returns a `Requestor`, and you can execute its `call()` or `acall()` to get synchronous or asynchronous API calls.

Synchronous API usage:

```python
from handyllm import OpenAIClient
with OpenAIClient(api_key='<your-key>') as client:
    response = client.chat(
      	model="gpt-4-turbo",
      	messages=[{"role": "user", "content": "please tell me a joke"}]
    ).call()  ## note .call() here
    print(response['choices'][0]['message']['content'])
```

Asynchronous API usage:

```python
async with OpenAIClient('async', api_key='<your-key>') as client_async:
    response = await client_async.chat(
      	model="gpt-4-turbo",
      	messages=[{"role": "user", "content": "please tell me a joke"}]
    ).acall()  ## note .acall() here
    print(response['choices'][0]['message']['content'])
```

You can instantiate a client that supports both modes:

```python
client = OpenAIClient('sync')  ## only supports sync APIs
client = OpenAIClient('async')  ## only supports async APIs
client = OpenAIClient('both')  ## supports both versions
```



### Legacy: Using OpenAIAPI proxy

> [!IMPORTANT]
> This is not recommended anymore. Use `OpenAIClient` instead.

Under the hood it connects to a module client and only provides **synchronous** APIs, **without** `call()`.

```python
from handyllm import OpenAIAPI
OpenAIAPI.api_key = '<your-key>'
response = OpenAIAPI.chat(
    model="gpt-4-turbo",
    messages=[{"role": "user", "content": "please tell me a joke"}]
)  ## no .call() here
print(response['choices'][0]['message']['content'])
```



## OpenAI API Request

### Endpoints

Each API request will connect to an endpoint along with some API configurations, which include:

|                  | Description                                                  | Value                   |
| ---------------- | ------------------------------------------------------------ | ----------------------- |
| api_type         | API type. Defaults to `openai`.                              | str: `openai` / `azure` |
| api_base         | API base url. Defaults to OpenAI base url.                   | str                     |
| api_key          | API key.                                                     | str                     |
| organization     | Organization.                                                | str                     |
| api_version      | API version. **Must be provided for Azure end-points.**      | str                     |
| model_engine_map | Map model name to engine name. Useful for Azure end-points if you have custom model names. | dict                    |

An `Endpoint` object contains these information. An `EndpointManager` acts like a list and can be used to rotate the next endpoint. See [test_endpoint.py](./tests/test_endpoint.py).

Methods for configuring endpoint info (values will be inferred in **top-town** order):

| Configuration method                               | Description                                                  |
| -------------------------------------------------- | ------------------------------------------------------------ |
| API keyword parameters                             | e.g.: `chat(api_key='xxx', ...)`                             |
| API `endpoint` keyword parameter                   | Providing an `Endpoint`, e.g.: `chat(endpoint=MyEndpoint)`   |
| API `endpoint_manager` keyword parameter           | Providing an `EndpointManager`, e.g.: `chat(endpoint_manager=MyEndpointManager)` |
| `OpenAIClient` instance (or `OpenAIAPI`) variables | e.g.: `client.api_key = 'xxx'` / `OpenAIAPI.api_key = 'xxx'` |
| Environment variables                              | `OPENAI_API_KEY`, `OPENAI_ORGANIZATION`/`OPENAI_ORG_ID`, `OPENAI_API_BASE`, `OPENAI_API_TYPE`, `OPENAI_API_VERSION`, `MODEL_ENGINE_MAP`. |

> [!TIP]
>
> **Azure OpenAI APIs are supported:** Specify `api_type='azure'`, and set `api_base` and `api_key` accordingly. Set `model_engine_map` if you want to use `model` parameter instead of `engine`/`deployment_id`. See [test_azure.py](./tests/test_azure.py). Please refer to [Azure OpenAI Service Documentation](https://learn.microsoft.com/en-us/azure/ai-services/openai/) for details.

### Logger

You can pass custom `logger` and `log_marks` (a string or a collection of strings) to `chat`/`completions` to get input and output logging.

### Timeout control

This toolkit supports client-side `timeout` control:

```python
from handyllm import OpenAIClient
client = OpenAIClient()
prompt = [{
    "role": "user",
    "content": "please tell me a joke"
    }]
response = client.chat(
    model="gpt-3.5-turbo",
    messages=prompt,
    timeout=10
    ).call()
print(response['choices'][0]['message']['content'])
```

### Stream response

Stream response of `chat`/`completions`/`finetunes_list_events` can be achieved using `steam` parameter:

```python
from handyllm import OpenAIClient, stream_chat

client = OpenAIClient()
response = client.chat(
    model="gpt-3.5-turbo",
    messages=prompt,
    timeout=10,
    stream=True
    ).call()

# you can use this to stream the response text
for text in stream_chat(response):
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



## Chat Prompt

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

> [!IMPORTANT]
>
> About the prompt format, each role key (e.g. `$system$` / `$user$` / `$assistant`) should be placed in a separate line.

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

