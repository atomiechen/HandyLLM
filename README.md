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

This toolkit uses HTTP API request instead of OpenAI's official python package to support client-side `timeout` control:

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

API key and organization will be loaded using the environment variable `OPENAI_API_KEY` and `OPENAI_ORGANIZATION`, or you can set manually:

```python
OpenAIAPI.api_key = 'sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
OpenAIAPI.organization = '......'  # default: None
```

Stream response of `chat`/`completions` can be achieved using `steam` parameter:

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



## Prompt

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
%misc%
```

```python
from handyllm import PromptConverter
converter = PromptConverter()

# chat can be used as the message parameter for OpenAI API
chat = converter.rawfile2chat('prompt.txt')

# variables wrapped in %s can be replaced at runtime
new_chat = converter.chat_replace_variables(chat, {r'%misc%': 'Note: do not use any bad word.'})
```



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

