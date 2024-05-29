# Change Log

All notable changes to HandyLLM will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).


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


