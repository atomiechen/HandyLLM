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
