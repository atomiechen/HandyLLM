from __future__ import annotations
__all__ = [
    "HandyPrompt",
    "ChatPrompt",
    "CompletionsPrompt",
    "loads",
    "load",
    "load_from",
    "dumps",
    "dump",
    "dump_to",
    "load_var_map",
    "RunConfig",
    "RecordRequestMode",
    "CredentialType",
]

import inspect
import json
import re
import copy
import io
import sys
from pathlib import Path
from datetime import datetime
from typing import IO, Any, MutableMapping, Optional, Union, TypeVar, cast
from abc import abstractmethod, ABC
from dataclasses import replace
from contextlib import contextmanager

import yaml
import frontmatter
from mergedeep import merge as merge_dict, Strategy
from dotenv import load_dotenv

from .prompt_converter import PromptConverter
from .openai_client import ClientMode, OpenAIClient
from .utils import (
    astream_chat_all, astream_completions, 
    stream_chat_all, stream_completions, 
)
from .run_config import RunConfig, RecordRequestMode, CredentialType
from ._types import PathType, SyncHandlerCompletions, VarMapType, SyncHandlerChat


PromptType = TypeVar('PromptType', bound='HandyPrompt')


converter = PromptConverter()
handler = frontmatter.YAMLHandler()
p_var_map = re.compile(r'(%\w+%)')

DEFAULT_CONFIG = RunConfig()
DEFAULT_BLACKLIST = (
    "api_key", "organization", "api_base", "api_type", "api_version", 
    "endpoint_manager", "endpoint", "engine", "deployment_id", 
    "model_engine_map", "dest_url", 
)


class HandyPrompt(ABC):
    
    TEMPLATE_OUTPUT_FILENAME = "result.%Y%m%d-%H%M%S.hprompt"
    TEMPLATE_OUTPUT_EVAL_FILENAME = "evaled.%Y%m%d-%H%M%S.hprompt"
    
    def __init__(
        self, 
        data: Union[str, list], 
        request: Optional[MutableMapping] = None, 
        meta: Optional[Union[MutableMapping, RunConfig]] = None, 
        base_path: Optional[PathType] = None,
        response: Optional[Any] = None,
        ):
        self.data = data
        self.request = request or {}
        # parse meta to run_config
        if isinstance(meta, RunConfig):
            self.run_config = meta
        else:
            self.run_config = RunConfig.from_dict(meta or {}, base_path=base_path)
        self.base_path = base_path
        self.response = response
    
    def __str__(self) -> str:
        return str(self.data)
    
    def __repr__(self) -> str:
        return "{}({}, {}, {})".format(
            type(self).__name__,
            repr(self.data),
            repr(self.request),
            repr(self.run_config)
        )
    
    @property
    def result_str(self) -> str:
        return str(self.data)
    
    def _serialize_data(self, data) -> str:
        '''
        Serialize the data to a string. 
        This method can be overridden by subclasses.
        '''
        return str(data)
    
    @classmethod
    def _dumps_frontmatter(cls, request: MutableMapping, run_config: RunConfig, base_path: Optional[PathType] = None) -> str:
        # dump frontmatter
        if not run_config and not request:
            return ""
        # need to filter the request
        front_data = cls._filter_request(request, run_config)
        if run_config:
            front_data['meta'] = run_config.to_dict(retain_object=False, base_path=base_path)
        post = frontmatter.Post("", None, **front_data)
        return frontmatter.dumps(post, handler).strip() + "\n\n"
    
    def dumps(self, base_path: Optional[PathType] = None) -> str:
        serialized_data = self._serialize_data(self.data)
        base_path = base_path or self.base_path
        return type(self)._dumps_frontmatter(self.request, self.run_config, base_path) + serialized_data
    
    def dump(self, fd: IO[str], base_path: Optional[PathType] = None) -> None:
        text = self.dumps(base_path=base_path)
        fd.write(text)
    
    def dump_to(self, path: PathType, mkdir: bool = False) -> None:
        if mkdir:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fd:
            self.dump(fd, base_path=Path(path).parent.resolve())
    
    @abstractmethod
    def _eval_data(self, run_config: RunConfig) -> Union[str, list]:
        ...
    
    def eval(
        self: PromptType, 
        run_config: RunConfig = DEFAULT_CONFIG,
        var_map: Optional[VarMapType] = None,
        **kwargs) -> PromptType:
        '''
        Evaluate the prompt with the given run_config. var_map is for convenience.
        A new prompt object is returned.
        '''
        new_run_config = self.eval_run_config(run_config)
        if var_map:
            if new_run_config.var_map is None:
                new_run_config.var_map = {}
            # merge var_map instead of replacing as a whole
            merge_dict(new_run_config.var_map, var_map, strategy=Strategy.REPLACE)
        new_data = self._eval_data(new_run_config)
        # update the request with the keyword arguments
        evaled_request = copy.deepcopy(self.request)
        evaled_request.update(kwargs)
        return type(self)(
            new_data,
            evaled_request,
            new_run_config,
        )
    
    def eval_run_config(
        self, 
        run_config: RunConfig, 
        ) -> RunConfig:
        # merge runtime run_config with the original run_config
        run_config = self.run_config.merge(run_config)
        
        cls = type(self)
        start_time = datetime.now()
        if run_config.output_path:
            run_config.output_path = cls._prepare_output_path(
                run_config.output_path, start_time, cls.TEMPLATE_OUTPUT_FILENAME
            )
        if run_config.output_evaled_prompt_path:
            run_config.output_evaled_prompt_path = cls._prepare_output_path(
                run_config.output_evaled_prompt_path, start_time, 
                cls.TEMPLATE_OUTPUT_EVAL_FILENAME
            )
        
        if run_config.credential_path:
            if not run_config.credential_type:
                # guess the credential type from the file extension
                p = Path(run_config.credential_path)
                if p.suffix:
                    run_config.credential_type = p.suffix[1:] # type: ignore
                else:
                    run_config.credential_type = CredentialType.ENV
        return run_config
    
    @classmethod
    @abstractmethod
    def _run_with_client(
        cls, 
        client: OpenAIClient, 
        evaled_prompt: PromptType,
        stream: bool,
        ) -> PromptType:
        ...
    
    def run(
        self: PromptType, 
        client: Optional[OpenAIClient] = None, 
        run_config: RunConfig = DEFAULT_CONFIG,
        var_map: Optional[VarMapType] = None,
        **kwargs) -> PromptType:
        evaled_prompt, stream = self._prepare_run(run_config, var_map, kwargs)
        cls = type(self)
        if client:
            new_prompt = cls._run_with_client(client, evaled_prompt, stream)
        else:
            with OpenAIClient(ClientMode.SYNC) as client:
                new_prompt = cls._run_with_client(client, evaled_prompt, stream)
        cls._post_check_output(stream, evaled_prompt.run_config, new_prompt)
        return new_prompt
    
    @classmethod
    @abstractmethod
    async def _arun_with_client(
        cls, 
        client: OpenAIClient, 
        evaled_prompt: PromptType,
        stream: bool,
        ) -> PromptType:
        ...
    
    async def arun(
        self: PromptType, 
        client: Optional[OpenAIClient] = None, 
        run_config: RunConfig = DEFAULT_CONFIG,
        var_map: Optional[VarMapType] = None,
        **kwargs) -> PromptType:
        evaled_prompt, stream = self._prepare_run(run_config, var_map, kwargs)
        cls = type(self)
        if client:
            new_prompt = await cls._arun_with_client(client, evaled_prompt, stream)
        else:
            async with OpenAIClient(ClientMode.ASYNC) as client:
                new_prompt = await cls._arun_with_client(client, evaled_prompt, stream)
        cls._post_check_output(stream, evaled_prompt.run_config, new_prompt)
        return new_prompt
    
    @staticmethod
    def _prepare_output_path(
        output_path: PathType, start_time: datetime, template_filename: str
        ) -> str:
        output_path = str(output_path).strip()
        p = Path(output_path)
        if p.is_dir() or output_path.endswith(('/')):
            # output_path wants to be a directory, append the default filename
            output_path = Path(output_path, template_filename)
        # format output_path with the current time
        output_path = start_time.strftime(str(output_path))
        return output_path
    
    def _prepare_run(self, run_config: RunConfig, var_map: Optional[MutableMapping], kwargs: MutableMapping):
        # evaluate the prompt with the given run_config
        evaled_prompt = self.eval(run_config=run_config, var_map=var_map, **kwargs)
        evaled_run_config = evaled_prompt.run_config
        evaled_request = evaled_prompt.request
        
        # get the stream flag
        stream = evaled_request.get("stream", False)
        
        # verbose output
        if evaled_run_config.verbose:
            print("---", file=sys.stderr)
            print("NEW RUN")
            print(f"Start time: {datetime.now()}", file=sys.stderr)
            evaled_run_config.pretty_print()
            print("---", file=sys.stderr)
        
        # output the evaluated prompt to a file or a file descriptor
        # NOTE: should be done before loading the credential file
        if evaled_run_config.output_evaled_prompt_fd:
            evaled_prompt.dump(evaled_run_config.output_evaled_prompt_fd)
        elif evaled_run_config.output_evaled_prompt_path:
            evaled_prompt.dump_to(evaled_run_config.output_evaled_prompt_path, mkdir=True)
        
        # load the credential file
        if evaled_run_config.credential_path:
            if evaled_run_config.credential_type == CredentialType.ENV:
                load_dotenv(evaled_run_config.credential_path, override=True)
            elif evaled_run_config.credential_type in (CredentialType.JSON, CredentialType.YAML):
                with open(evaled_run_config.credential_path, 'r', encoding='utf-8') as fin:
                    if evaled_run_config.credential_type == CredentialType.JSON:
                        credential_dict = json.load(fin)
                    else:
                        credential_dict = yaml.safe_load(fin)
                # do not overwrite the existing request arguments
                for key, value in credential_dict.items():
                    if key not in evaled_request:
                        evaled_request[key] = value
            else:
                raise ValueError(f"unsupported credential type: {evaled_run_config.credential_type}")
        
        return evaled_prompt, stream
    
    @staticmethod
    def _post_check_output(stream: bool, run_config: RunConfig, new_prompt: HandyPrompt):
        if not stream:
            # if stream is True, the response is already streamed to 
            # a file or a file descriptor
            if run_config.output_fd:
                new_prompt.dump(run_config.output_fd)
            elif run_config.output_path:
                new_prompt.dump_to(run_config.output_path, mkdir=True)
        return new_prompt

    def _merge_non_data(self: PromptType, other: PromptType, inplace=False) -> tuple[MutableMapping, RunConfig]:
        if inplace:
            merge_dict(self.request, other.request, strategy=Strategy.ADDITIVE)
            self.run_config.merge(other.run_config, inplace=True)
            return self.request, self.run_config
        else:
            merged_request = merge_dict({}, self.request, other.request, strategy=Strategy.ADDITIVE)
            merged_run_config = self.run_config.merge(other.run_config)
            return merged_request, merged_run_config
    
    @staticmethod
    def _filter_request(
        request: MutableMapping, 
        run_config: RunConfig,
        ) -> MutableMapping:
        if run_config.record_request == RecordRequestMode.WHITELIST:
            if run_config.record_whitelist:
                request = {key: value for key, value in request.items() if key in run_config.record_whitelist}
            else:
                request = {}
        elif run_config.record_request == RecordRequestMode.NONE:
            request = {}
        elif run_config.record_request == RecordRequestMode.ALL:
            pass
        else:
            # default: blacklist
            request = copy.deepcopy(request)
            real_blacklist = run_config.record_blacklist or DEFAULT_BLACKLIST
            for key in real_blacklist:
                request.pop(key, None)
        return request
    
    def _parse_var_map(self, run_config: RunConfig):
        var_map = {}
        if run_config.var_map_path:
            var_map = merge_dict(
                var_map, 
                load_var_map(run_config.var_map_path), 
                strategy=Strategy.REPLACE
            )
        if run_config.var_map:
            var_map = merge_dict(
                var_map, 
                run_config.var_map, 
                strategy=Strategy.REPLACE
            )
        return var_map

    @abstractmethod
    def __add__(self: PromptType, other) -> PromptType:
        ...

    @staticmethod
    @contextmanager
    def open_output_path_fd(run_config: RunConfig):
        run_config.output_path = cast(PathType, run_config.output_path)
        # create the parent directory if it does not exist
        Path(run_config.output_path).parent.mkdir(parents=True, exist_ok=True)
        if run_config.output_path_buffering is None or run_config.output_path_buffering == -1:
            # default buffering
            with open(run_config.output_path, 'w', encoding='utf-8') as fout:
                yield fout
        elif run_config.output_path_buffering == 0:
            # no buffering
            with open(run_config.output_path, 'wb', buffering=0) as f_binary:
                yield io.TextIOWrapper(f_binary, encoding='utf-8', write_through=True)
        elif isinstance(run_config.output_path_buffering, int) and run_config.output_path_buffering >= 1:
            # 1 for line buffering, >= 2 for buffer size
            with open(run_config.output_path, 'w', encoding='utf-8', buffering=run_config.output_path_buffering) as fout:
                fout.reconfigure(write_through=True)
                yield fout
        else:
            raise ValueError(f"unsupported output_path_buffering value: {run_config.output_path_buffering}")


class ChatPrompt(HandyPrompt):
        
    def __init__(
        self, 
        messages: list, 
        request: Optional[MutableMapping] = None, 
        meta: Optional[Union[MutableMapping, RunConfig]] = None, 
        base_path: Optional[PathType] = None,
        response: Optional[Any] = None,
        ):
        super().__init__(messages, request, meta, base_path, response)
    
    @property
    def messages(self) -> list:
        return self.data
    
    @messages.setter
    def messages(self, value: list):
        self.data = value
    
    @property
    def result_str(self) -> str:
        if len(self.messages) == 0:
            return ""
        return self.messages[-1]['content']
    
    def _serialize_data(self, data) -> str:
        return converter.msgs2raw(data)
    
    def _eval_data(self, run_config: RunConfig) -> list:
        var_map = self._parse_var_map(run_config)
        return converter.msgs_replace_variables(
            self.messages, var_map, inplace=False)
    
    @staticmethod
    def _wrap_gen_chat(response, run_config: RunConfig):
        for role, content, tool_call in stream_chat_all(response):
            if run_config.on_chunk:
                run_config.on_chunk = cast(SyncHandlerChat, run_config.on_chunk)
                run_config.on_chunk(role, content, tool_call)
            yield role, content, tool_call
    
    @staticmethod
    async def _awrap_gen_chat(response, run_config: RunConfig):
        async for role, content, tool_call in astream_chat_all(response):
            if run_config.on_chunk:
                if inspect.iscoroutinefunction(run_config.on_chunk):
                    await run_config.on_chunk(role, content, tool_call)
                else:
                    run_config.on_chunk = cast(SyncHandlerChat, run_config.on_chunk)
                    run_config.on_chunk(role, content, tool_call)
            yield role, content, tool_call

    @classmethod
    def _run_with_client(
        cls, 
        client: OpenAIClient, 
        evaled_prompt: HandyPrompt,
        stream: bool,
        ) -> ChatPrompt:
        run_config = evaled_prompt.run_config
        new_request = evaled_prompt.request
        response = client.chat(
            messages=evaled_prompt.data,
            **new_request
            ).call()
        base_path = Path(run_config.output_path).parent.resolve() if run_config.output_path else None
        if stream:
            if run_config.output_fd:
                # dump frontmatter, no base_path
                run_config.output_fd.write(cls._dumps_frontmatter(new_request, run_config))
                # stream response to a file descriptor
                role, content, tool_calls = converter.stream_msgs2raw(
                    cls._wrap_gen_chat(response, run_config), 
                    run_config.output_fd
                    )
            elif run_config.output_path:
                # stream response to a file
                with cls.open_output_path_fd(run_config) as fout:
                    # dump frontmatter
                    fout.write(cls._dumps_frontmatter(new_request, run_config, base_path))
                    role, content, tool_calls = converter.stream_msgs2raw(
                        cls._wrap_gen_chat(response, run_config), 
                        fout
                        )
            else:
                role, content, tool_calls = converter.stream_msgs2raw(
                    cls._wrap_gen_chat(response, run_config)
                    )
        else:
            response = cast(Any, response)
            role = response['choices'][0]['message']['role']
            content = response['choices'][0]['message'].get('content')
            tool_calls = response['choices'][0]['message'].get('tool_calls')
        return ChatPrompt(
            [{"role": role, "content": content, "tool_calls": tool_calls}],
            new_request, run_config, base_path,
            response=response
        )
    
    @classmethod
    async def _arun_with_client(
        cls, 
        client: OpenAIClient, 
        evaled_prompt: HandyPrompt,
        stream: bool,
        ) -> ChatPrompt:
        run_config = evaled_prompt.run_config
        new_request = evaled_prompt.request
        response = await client.chat(
            messages=evaled_prompt.data,
            **new_request
            ).acall()
        base_path = Path(run_config.output_path).parent.resolve() if run_config.output_path else None
        if stream:
            if run_config.output_fd:
                # stream response to a file descriptor
                run_config.output_fd.write(cls._dumps_frontmatter(new_request, run_config))
                role, content, tool_calls = await converter.astream_msgs2raw(
                    cls._awrap_gen_chat(response, run_config), 
                    run_config.output_fd
                    )
            elif run_config.output_path:
                # stream response to a file
                with cls.open_output_path_fd(run_config) as fout:
                    fout.write(cls._dumps_frontmatter(new_request, run_config, base_path))
                    role, content, tool_calls = await converter.astream_msgs2raw(
                        cls._awrap_gen_chat(response, run_config), 
                        fout
                        )
            else:
                role, content, tool_calls = await converter.astream_msgs2raw(
                    cls._awrap_gen_chat(response, run_config)
                    )
        else:
            response = cast(Any, response)
            role = response['choices'][0]['message']['role']
            content = response['choices'][0]['message'].get('content')
            tool_calls = response['choices'][0]['message'].get('tool_calls')
        return ChatPrompt(
            [{"role": role, "content": content, "tool_calls": tool_calls}],
            new_request, run_config, base_path,
            response=response
        )

    def __add__(self: ChatPrompt, other: Union[str, dict, list, ChatPrompt]):
        # support concatenation with string, list, dict or another ChatPrompt
        new_prompt = copy.deepcopy(self)
        new_prompt += other
        return new_prompt
    
    def __iadd__(self: ChatPrompt, other: Union[str, dict, list, ChatPrompt]):
        # support concatenation with string, list, dict or another ChatPrompt
        if isinstance(other, str):
            self.add_message(content=other)
        elif isinstance(other, dict):
            self.messages.append(other)
        elif isinstance(other, list):
            for item in other:
                self.messages.append(item)
        elif isinstance(other, ChatPrompt):
            # merge two ChatPrompt objects
            self += other.messages
            self._merge_non_data(other, inplace=True)
        else:
            raise TypeError(f"unsupported operand type(s) for +: 'ChatPrompt' and '{type(other)}'")
        return self
    
    def add_message(self, role: str = "user", content: Optional[str] = None, tool_calls: Optional[list] = None):
        msg = {"role": role, "content": content}
        if tool_calls is not None:
            msg["tool_calls"] = tool_calls
        self.messages.append(msg)


class CompletionsPrompt(HandyPrompt):
    
    def __init__(
        self, 
        prompt: str, 
        request: Optional[MutableMapping] = None, 
        meta: Optional[Union[MutableMapping, RunConfig]] = None, 
        base_path: Optional[PathType] = None,
        response: Optional[Any] = None,
        ):
        super().__init__(prompt, request, meta, base_path, response)
    
    @property
    def prompt(self) -> str:
        return self.data
    
    @prompt.setter
    def prompt(self, value: str):
        self.data = value
    
    def _eval_data(self, run_config: RunConfig) -> str:
        var_map = self._parse_var_map(run_config)
        new_prompt = self.prompt
        for key, value in var_map.items():
            new_prompt = new_prompt.replace(key, value)
        return new_prompt
    
    @staticmethod
    def _stream_completions_proc(response, run_config: RunConfig, fd: Optional[IO[str]] = None) -> str:
        # stream response to fd
        content = ""
        for text in stream_completions(response):
            if run_config.on_chunk:
                run_config.on_chunk = cast(SyncHandlerCompletions, run_config.on_chunk)
                run_config.on_chunk(text)
            if fd:
                fd.write(text)
            content += text
        return content

    @classmethod
    def _run_with_client(
        cls, 
        client: OpenAIClient, 
        evaled_prompt: HandyPrompt,
        stream: bool,
        ) -> CompletionsPrompt:
        run_config = evaled_prompt.run_config
        new_request = evaled_prompt.request
        response = client.completions(
            prompt=evaled_prompt.data,
            **new_request
            ).call()
        base_path = Path(run_config.output_path).parent.resolve() if run_config.output_path else None
        if stream:
            if run_config.output_fd:
                # stream response to a file descriptor
                run_config.output_fd.write(cls._dumps_frontmatter(new_request, run_config))
                content = cls._stream_completions_proc(response, run_config, run_config.output_fd)
            elif run_config.output_path:
                # stream response to a file
                with cls.open_output_path_fd(run_config) as fout:
                    fout.write(cls._dumps_frontmatter(new_request, run_config, base_path))
                    content = cls._stream_completions_proc(response, run_config, fout)
            else:
                content = cls._stream_completions_proc(response, run_config)
        else:
            response = cast(Any, response)
            content = response['choices'][0]['text']
        return CompletionsPrompt(
            content, new_request, run_config, base_path, response=response
            )

    @staticmethod
    async def _astream_completions_proc(response, run_config: RunConfig, fd: Optional[IO[str]] = None) -> str:
        # stream response to fd
        content = ""
        async for text in astream_completions(response):
            if run_config.on_chunk:
                if inspect.iscoroutinefunction(run_config.on_chunk):
                    await run_config.on_chunk(text)
                else:
                    run_config.on_chunk = cast(SyncHandlerCompletions, run_config.on_chunk)
                    run_config.on_chunk(text)
            if fd:
                fd.write(text)
            content += text
        return content
    
    @classmethod
    async def _arun_with_client(
        cls, 
        client: OpenAIClient, 
        evaled_prompt: HandyPrompt,
        stream: bool,
        ) -> CompletionsPrompt:
        run_config = evaled_prompt.run_config
        new_request = evaled_prompt.request
        response = await client.completions(
            prompt=evaled_prompt.data,
            **new_request
            ).acall()
        base_path = Path(run_config.output_path).parent.resolve() if run_config.output_path else None
        if stream:
            if run_config.output_fd:
                # stream response to a file descriptor
                run_config.output_fd.write(cls._dumps_frontmatter(new_request, run_config))
                content = await cls._astream_completions_proc(response, run_config, run_config.output_fd)
            elif run_config.output_path:
                # stream response to a file
                with cls.open_output_path_fd(run_config) as fout:
                    fout.write(cls._dumps_frontmatter(new_request, run_config, base_path))
                    content = await cls._astream_completions_proc(response, run_config, fout)
            else:
                content = await cls._astream_completions_proc(response, run_config)
        else:
            response = cast(Any, response)
            content = response['choices'][0]['text']
        return CompletionsPrompt(
            content, new_request, run_config, base_path, response=response
            )
    
    def __add__(self: CompletionsPrompt, other: Union[str, CompletionsPrompt]):
        # support concatenation with string or another CompletionsPrompt
        new_prompt = copy.deepcopy(self)
        new_prompt += other
        return new_prompt
    
    def __iadd__(self: CompletionsPrompt, other: Union[str, CompletionsPrompt]):
        # support concatenation with string or another CompletionsPrompt
        if isinstance(other, str):
            self.add_text(other)
        elif isinstance(other, CompletionsPrompt):
            # merge two CompletionsPrompt objects
            self.add_text(other.prompt)
            self._merge_non_data(other, inplace=True)
        else:
            raise TypeError(f"unsupported operand type(s) for +: 'CompletionsPrompt' and '{type(other)}'")
        return self

    def add_text(self, text: str):
        self.prompt += text


def loads(
    text: str, 
    encoding: str = "utf-8",
    base_path: Optional[PathType] = None,
    cls: type[PromptType] = HandyPrompt,
) -> PromptType:
    if handler.detect(text):
        metadata, data = frontmatter.parse(text, encoding, handler)
        meta = metadata.pop("meta", None)
        if not isinstance(meta, dict):
            meta = {}
        request = metadata
    else:
        data = text
        request = {}
        meta = {}
    if cls == HandyPrompt:
        # get specific prompt class
        api: str = meta.get("api", "")
        if api:
            api = api.lower()
            if api.startswith("chat"):
                cls = cast(type[PromptType], ChatPrompt)
            else:
                cls = cast(type[PromptType], CompletionsPrompt)
        else:
            if converter.detect(data):
                cls = cast(type[PromptType], ChatPrompt)
            else:
                cls = cast(type[PromptType], CompletionsPrompt)
    if cls == ChatPrompt:
        data = converter.raw2msgs(data)
    return cls(data, request, meta, base_path)


def load(
    fd: IO[str], 
    encoding: str = "utf-8",
    base_path: Optional[PathType] = None,
    cls: type[PromptType] = HandyPrompt,
) -> PromptType:
    text = fd.read()
    return loads(text, encoding, base_path=base_path, cls=cls)

def load_from(
    path: PathType,
    encoding: str = "utf-8",
    cls: type[PromptType] = HandyPrompt,
) -> PromptType:
    with open(path, "r", encoding=encoding) as fd:
        return load(fd, encoding, base_path=Path(path).parent.resolve(), cls=cls)

def dumps(
    prompt: HandyPrompt, 
    base_path: Optional[PathType] = None
) -> str:
    return prompt.dumps(base_path)

def dump(
    prompt: HandyPrompt, 
    fd: IO[str], 
    base_path: Optional[PathType] = None
) -> None:
    return prompt.dump(fd, base_path)

def dump_to(
    prompt: HandyPrompt, 
    path: PathType, 
    mkdir: bool = False, 
) -> None:
    return prompt.dump_to(path, mkdir=mkdir)

def load_var_map(path: PathType) -> dict[str, str]:
    # read all content that needs to be replaced in the prompt from a text file
    with open(path, 'r', encoding='utf-8') as fin:
        content = fin.read()
    substitute_map = {}
    blocks = p_var_map.split(content)
    for idx in range(1, len(blocks), 2):
        key = blocks[idx]
        value = blocks[idx+1]
        substitute_map[key] = value.strip()
    return substitute_map
