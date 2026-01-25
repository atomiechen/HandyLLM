# Change Log

All notable changes to HandyLLM will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).


## [0.10.0] - 2026-01-25

### Added

- **Reasoning support** :
  - support for reasoning sub-headers in hprompt files (`$$reasoning_content$$` / `$$content$$`) to store reasoning tokens separately from visible content.
  - add `result_reasoning` on `ChatPrompt` to retrieve the reasoning content from the last message.
  - add `stream_chat_with_reasoning` and `astream_chat_with_reasoning` for streaming responses that include reasoning tokens.
- `PromptConverter`: introduce `tool` and `array` keywords for dumped `ChatPrompt`.
- `hprompt`:
  - store the `requestor` on the resulting prompt.
  - support `input_audio` and `file` inputs, and also support loading from local paths.
  - `ChatPrompt.add_message()` now suggests role literal types and accepts a `tool_call_id`.
  - add methods to append content parts to an existing message (text, image, audio, file).
- `Requestor`: store `api_base` and `request_url`, and compute full `url` on demand.
- `OpenAIClient`: support `ensure_client_credentials` in the constructor and loaders to enforce client-wide credentials (ignore runtime API creds). Default is `False` for backward compatibility.
- `RunConfig`: add `overwrite_if_exists`. Default is `False` to prevent overwriting existing output files, and will append suffixes if the file exists.
- `CacheManager`: add `only_load` option and an `ensure_dumped()` helper to avoid repeated loads when checking files.
  - Example: `ensure_dumped()` helps when a cached function (using `cache`) may be invoked multiple times but you only want to verify file existence without reloading the content.

### Fixed

- `PromptConverter`:
  - escape quotes/newlines in headers.
  - ensure `msgs_replace_variables()` returns a deep copy when `inplace=False`. The previous behavior returns a shallow list copy, which may cause unexpected behavior when modifying the returned object.
- `ChatPrompt`: `result_str` always returns a `str`; if the last message content is a list, the text part is returned when present.

### Changed

- Global hprompt loaders now default to `ChatPrompt`. `HandyPrompt.load_from()`/`.load()`/`.loads()` will auto-detect prompt type; call `ChatPrompt.load_from()` or `CompletionsPrompt.load_from()` to explicitly request a type.
- `PromptConverter.msgs2raw()` (ChatPrompt dump) now uses `tool` and `array` keywords by default (replacing the previous `type="tool_calls"` and `type="content_array"` markers).
- Streaming chunk contract: `RunConfig.on_chunk()` now receives a typed dict `ChatChunkUnified` with keys `role`, `content`, `reasoning_content`, and `tool_call`; `stream_chat_all()` / `astream_chat_all()` yield the same structure.
- Replace `DictProxy` usages with `TypedDict` for clearer typing and editor support.


## [0.9.3] - 2024-12-18

### Fixed

- Due to the change in `httpx==0.28.0` (see its [CHANGELOG](https://github.com/encode/httpx/blob/master/CHANGELOG.md)), the request `params` cannot be `{}` as it will override the original query params. This will cause errors like 404 Resource Not Found, when the `api-version` parameter in Azure endpoint is overridden. Now when `params` is an empty dictionary, it will be passed as the default `None`. See httpx [PR 3364](https://github.com/encode/httpx/pull/3364) and [Issue 3433](https://github.com/encode/httpx/issues/3433). 


## [0.9.2] - 2024-08-09

### Fixed

- fix DictProxy YAML dump
- fix tool calls in stream mode: missing last tool call

### Changed

- change import behavior:
  - `__init__.py`: 
    - use explicit imports instead of `*`
    - limit imports from `utils.py`
    - remove all imports from `response.py` and `types.py` (please import them directly)
  - `response.py`: add missing items to `__all__`
- add `py.typed` for better IDE support


## [0.9.1] - 2024-07-28

### Added

- 🔧 add lint dependency ruff
  - add `scripts/lint.sh` for linting
  - add `scripts/format.sh` for formatting, and format all files
  - add CI test workflow

### Fixed

- fix errors reported by ruff (E711, E722, F541)
- fix TypeError on python 3.8 ('type' object is not subscriptable)


## [0.9.0] - 2024-07-28

### Added

Check 🌟s for attractive new features!

- `hprompt.py`: 
  - 🌟 `image_url` in chat hprompt file now supports local path (file://), both absolute and relative
  - 🌟 add `fetch()`, `afetch()`, `stream()` and `astream()` methods for direct and typed API responses
  - add `RunConfig.var_map_file_format` for specifying variable map file format, including JSON / YAML; `load_var_map()` supports format param
- `requestor.py`: 
  - 🌟 add `fetch()`, `afetch()`, `stream()` and `astream()` methods for typed API responses
  - use generic and add `DictRequestor`, `BinRequestor`, `ChatRequestor` and `CompletionsRequestor`
- `OpenAIClient`: 
  - 🌟 constructor supports `endpoint_manager`, `endpoints` and `load_path` param; supports loading from YAML file and `Mapping` obj
  - APIs support `endpoints` param
  - APIs `endpoint` param supports `Mapping` type
- 🌟 added `cache_manager.py`: `CacheManager` for general purpose caching to text files
  - add `load_method` and `dump_method` params
  - infers format from file suffix when convert handler is not provided
- 🌟 added `response.py`: `DictProxy` for quick access to well-defined response dict
- `EndpointManager`: supports loading from YAML file using `endpoints` key, or from `Iterable` obj
- `__init__.py` import everything from hprompt for convenience
- rename `_types.py` to `types.py` and expose all definitions
- `prompt_converter.py`:
  - add generator sink `consume_stream2fd()`
- `utils.py`:
  - add generator filter `trans_stream_chat()`, generator sink `echo_consumer()`
- a lot improved type hints
- added tests:
  - load prompt type specification
  - variable map substitution
  - ChatPrompt and CompletionsPrompt's API calls, supports for RunConfig.on_chunk, and addition operations
  - chat hprompt `image_url`
  - `OpenAIClient` loading, chat `fetch()` & `stream()`
  - `endpoint_manager.py`
  - `cache_manager.py`
  - audio speech
  - legacy OpenAIAPI

### Changed

- `hprompt.py`: 
  - add 'endpoints' to default record blacklist
  - remove the `var_map` related configurations from the evaluated prompt, as it is already applied
- `EndpointManager`: 
  - raises ValueError when getting endpoint out of empty

### Fixed

- `hprompt.py`: 'type' object is not subscriptable on python 3.8

### Removed

- `prompt_converter.py`: remove `stream_msgs2raw()` and `astream_msgs2raw()` as no longer needed


## [0.8.2] - 2024-06-30

### Added

- `hprompt`: load methods now support `cls` parameter for prompt type specification
- `ChatPrompt` and `CompletionsPrompt` support optional request and meta
- `ChatPrompt` :
  - supports add dict
  - add `add_message(...)` method
- `CompletionsPrompt`:
  - add `add_text(...)` method
- `PromptConverter`: `yaml.dump` uses `allow_unicode=True` option
- move all type definitions to `_types.py`
- support for package development:
  - add `requirement.txt` for development
  - add `scripts/test.sh` for running tests
  - add test scripts in `tests` folder

### Fixed

- `HandyPrompt.eval(...)` should not make directories for output paths
- `CompletionsPrompt._run_with_client(...)`: misplaced `run_config` param
- `PromptConverter`
  - fix variable replacement for `content_array` message
  - fix wrong return type of `stream_msgs2raw` and `astream_msgs2raw`
- `requestor`:
  - `httpx.Response` should use `reason_phrase` to get error reason
  - `acall()` fix missing brackets for await
  - `_call_raw()` and `_acall_raw()` intercept and raise new exception without original one
  - `_acall_raw()`: read the response first to prevent `httpx.ResponseNotRead` before getting error message
- `_utils.exception2err_msg(...)` should append error message instead of printing
- change `io.IOBase` to `IO[str]` for file descriptors (e.g. `RunConfig.output_fd`)
- fix other type hints

### Changed

- move all old files in `tests` folder to `examples` folder


## [0.8.1] - 2024-06-10

### Fixed

- fix the debug print issue when outputting to a file in stream mode


## [0.8.0] - 2024-06-09

### Added

- CLI: output to stderr without buffering
- add `RunConfig.output_path_buffering` for controlling buffering of output file
- add this changelog

### Fixed

- fix `_post_check_output(...)` not using evaluated `run_config` (may cause `output_path` or `output_fd` to be ignored)

### Changed

- rename internal constants to remove leading `_` of `API_xxx` constants

### Removed

- remove unused files in `deprecated` folder


## [0.7.6] - 2024-05-24

### Added

- add `RunConfig.on_chunk` as callback for streamed chunks
- add Azure tts example
- add `VM` method to transform kwargs to % wrapped variable map dict
- add `var_map` arg to `eval(...)`, `run(...)` and `arun(...)` for convenience

### Changed

- merging different versions of `var_map` from method argument or from another `RunConfig`, instead of replacing it as a whole
- rename `RunConfig.to_dict`'s `retain_fd` arg to `retain_object`


## [0.7.5] - 2024-05-23

### Added

- `OpenAIClient` add audio speech (tts) api support
  - add azure support for audio speech and transcriptions
- add tts test script

### Changed

- prioritize `RunConfig.output_evaled_prompt_fd` over `RunConfig.output_evaled_prompt_path`
- `eval(...)`
  - always return a new object
  - gives `run_config` arg a default value
  - accepts kwargs, same as `run(...)`
- when dumping, always filter request
- credential file do not overwrite existing request args

### Fixed

- non-stream mode prioritize `RunConfig.output_fd` over `RunConfig.output_path`


