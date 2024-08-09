# 快速上手

以下是一个最小hprompt文件示例：

````hprompt
{! ../docs_src/mwe.hprompt !}
````

运行它：

```sh
handyllm hprompt try.hprompt
```

结果会输出到标准错误，格式仍然是hprompt。

你也可以在代码中运行：

```python
from handyllm import hprompt
my_prompt = hprompt.load_from('try.hprompt')
result_prompt = my_prompt.run()
print(result_prompt.dumps())
```

### 更多控制

你可以在frontmatter中加入更多参数，或者在内容中添加变量：

```hprompt
---
# frontmatter data
model: gpt-3.5-turbo
temperature: 0.5
meta:
  credential_path: .env
  var_map_path: substitute.txt
  output_path: out/%Y-%m-%d/result.%H-%M-%S.hprompt
  output_evaled_prompt_path: out/%Y-%m-%d/evaled.%H-%M-%S.hprompt
---

$system$
You are a helpful assistant.

$user$
Your current context: 
%context%

Please follow my instructions:
%instructions%
```

## 变量

运行前，类似于 `%content%` 的变量会由 `meta.var_map_path` 文件中的内容替换，或者被字典 `meta.var_map` 替换。

如果想要在代码中替换，可传入运行时的 `var_map` 参数：

```python
from handyllm import hprompt, VM
...
result_prompt = my_prompt.run(var_map=VM(
    context='It is raining outside.',
    instructions='Write a poem.'
))
```

## 工具调用

首先在frontmatter中指定 `tools`：

````hprompt
{! ../docs_src/tool.hprompt [ln:1-22] !}
````

然后运行hprompt等待结果，此时assistant返回的结果将会是tool类型、以YAML格式展示：

````hprompt
{! ../docs_src/tool.hprompt [ln:24-39] !}
````

如果要进一步调用，则需要先按id给出各个工具调用的结果，与原messages拼接：

````hprompt
{! ../docs_src/tool.hprompt [ln:42-46] !}
````

上述内容的完整hprompt：

````hprompt
{! ../docs_src/tool.hprompt !}
````

## 图片输入

首先需要在 `$user$` 中指定为 `array` 类型 （或者写为 `type="content_array"`），表示输入的部分为YAML列表：

````hprompt
{! ../docs_src/image.hprompt !}
````

HandyLLM 同样支持直接从本地文件或base64字符串加载图片：

````hprompt
{! ../docs_src/image_local.hprompt !}
````

````hprompt
{! ../docs_src/image_base64.hprompt !}
````
