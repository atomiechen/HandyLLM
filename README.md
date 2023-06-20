# HandyLLM

A handy toolkit for using LLM.



## OpenAI API Request

This toolkit uses HTTP API request instead of OpenAI's official python package to support client-side `timeout` control.

```python
from handyllm import OpenAIAPI
OpenAIAPI.api_key = os.environ.get('OPENAI_API_KEY')
response = OpenAIAPI.chat(
    model="gpt-3.5-turbo",
    messages=prompt,
    timeout=10
    )
```



## Prompt

`PromptConverter` can convert this text file into a structured prompt for chat API calls.

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
```

It can also help substitute placeholder variables (e.g. `%output_format%` in this example) to make multiple prompts modular. A substitute map looks like this:

```
%output_format%
Please output a SINGLE JSON object that contains all items from the two input JSON objects.

%variable1%
Placeholder text.

%variable2%
Placeholder text.

%variable3%
Placeholder text.
```

