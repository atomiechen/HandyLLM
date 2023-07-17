# HandyLLM

[![PyPI](https://img.shields.io/pypi/v/HandyLLM)](https://github.com/atomiechen/HandyLLM) ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/Handyllm)

A handy toolkit for using LLM.



## Install

```shell
pip3 install handyllm
```



## OpenAI API Request

This toolkit uses HTTP API request instead of OpenAI's official python package to support client-side `timeout` control:

```python
from handyllm import OpenAIAPI
OpenAIAPI.api_key = os.environ.get('OPENAI_API_KEY')
response = OpenAIAPI.chat(
    model="gpt-3.5-turbo",
    messages=prompt,
    timeout=10
    )
```

API key will be loaded using the environment variable `OPENAI_API_KEY`, or you can set manually:

```python
OpenAIAPI.api_key = 'sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
OpenAIAPI.organization = '......'  # default to None
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

