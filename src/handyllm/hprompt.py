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
    "RequestRecordMode",
]

import re
import copy
import io
import os
from typing import Optional, Union, TypeVar
from enum import Enum, auto
from abc import abstractmethod, ABC
from dataclasses import dataclass

import frontmatter
from mergedeep import merge, Strategy

from .prompt_converter import PromptConverter
from .openai_client import ClientMode, OpenAIClient
from .utils import (
    astream_chat_with_role, astream_completions, 
    stream_chat_with_role, stream_completions, 
)


PromptType = TypeVar('PromptType', bound='HandyPrompt')
PathType = Union[str, os.PathLike[str]]

converter = PromptConverter()
handler = frontmatter.YAMLHandler()
p_var_map = re.compile(r'(%\w+%)')

DEFAULT_BLACKLIST = (
    "api_key", "organization", "api_base", "api_type", "api_version", 
    "endpoint_manager", "endpoint", "engine", "deployment_id", 
    "model_engine_map", "dest_url", 
)


def loads(
    text: str, 
    encoding: str = "utf-8"
) -> HandyPrompt:
    if handler.detect(text):
        metadata, content = frontmatter.parse(text, encoding, handler)
        meta = metadata.pop("meta", None) or {}
        request = metadata
    else:
        content = text
        request = {}
        meta = {}
    api: str = meta.get("api", "")
    is_chat = converter.detect(content)
    if api.startswith("completion") or not is_chat:
        return CompletionsPrompt(content, request, meta)
    else:
        chat = converter.raw2chat(content)
        return ChatPrompt(chat, request, meta)

def load(
    fd: io.IOBase, 
    encoding: str = "utf-8"
) -> HandyPrompt:
    text = fd.read()
    return loads(text, encoding)

def load_from(
    path: PathType,
    encoding: str = "utf-8"
) -> HandyPrompt:
    with open(path, "r", encoding=encoding) as fd:
        return load(fd, encoding)

def dumps(
    prompt: HandyPrompt, 
) -> str:
    return prompt.dumps()

def dump(
    prompt: HandyPrompt, 
    fd: io.IOBase, 
) -> None:
    return prompt.dump(fd)

def dump_to(
    prompt: HandyPrompt, 
    path: PathType
) -> None:
    return prompt.dump_to(path)

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


class RequestRecordMode(Enum):
    BLACKLIST = auto()  # record all request arguments except specified ones
    WHITELIST = auto()  # record only specified request arguments
    NONE = auto()  # record no request arguments
    ALL = auto()  # record all request arguments


@dataclass
class RunConfig:
    record: RequestRecordMode = RequestRecordMode.BLACKLIST
    blacklist: Optional[list[str]] = DEFAULT_BLACKLIST
    whitelist: Optional[list[str]] = None
    # variable map
    var_map: Optional[dict[str, str]] = None
    # variable map file path
    var_map_path: Optional[PathType] = None
    # output the result to a file or a file descriptor
    output_path: Optional[PathType] = None
    output_fd: Optional[io.IOBase] = None
    # output the evaluated prompt to a file or a file descriptor
    output_evaled_prompt_path: bool = False
    output_evaled_prompt_fd: Optional[io.IOBase] = None


DEFAULT_CONFIG = RunConfig()


class HandyPrompt(ABC):
    
    def __init__(self, data: Union[str, list], request: dict = None, meta: dict = None):
        self.data = data
        self.request = request or {}
        self.meta = meta or {}
    
    def __str__(self) -> str:
        return str(self.data)
    
    def __repr__(self) -> str:
        return "{}({}, {}, {})".format(
            self.__class__.__name__,
            repr(self.data),
            repr(self.request),
            repr(self.meta)
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
    
    @staticmethod
    def _dumps(request, meta, content: str) -> str:
        if not meta and not request:
            return content
        front_data = copy.deepcopy(request)
        if meta:
            front_data['meta'] = copy.deepcopy(meta)
        post = frontmatter.Post(content, None, **front_data)
        return frontmatter.dumps(post, handler)
    
    def dumps(self) -> str:
        serialized_data = self._serialize_data(self.data)
        return self._dumps(self.request, self.meta, serialized_data)
    
    def dump(self, fd: io.IOBase) -> None:
        text = self.dumps()
        fd.write(text)
    
    def dump_to(self, path: PathType) -> None:
        with open(path, "w", encoding="utf-8") as fd:
            self.dump(fd)
    
    @abstractmethod
    def _eval_data(self: PromptType, run_config: RunConfig) -> Union[str, list]:
        ...
    
    def eval(self: PromptType, run_config: RunConfig) -> PromptType:
        new_data = self._eval_data(run_config)
        if new_data != self.data:
            return self.__class__(
                new_data,
                copy.deepcopy(self.request),
                copy.deepcopy(self.meta)
            )
        return self
    
    @abstractmethod
    def _run_with_client(
        self: PromptType, 
        client: OpenAIClient, 
        run_config: RunConfig,
        new_request: dict,
        new_meta: dict,
        stream: bool,
        ) -> PromptType:
        ...
    
    def run(
        self: PromptType, 
        client: OpenAIClient = None, 
        run_config: RunConfig = DEFAULT_CONFIG,
        **kwargs) -> PromptType:
        run_config, new_request, new_meta, stream = self._prepare_run(run_config, kwargs)
        if client:
            new_prompt = self._run_with_client(client, run_config, new_request, new_meta, stream)
        else:
            with OpenAIClient(ClientMode.SYNC) as client:
                new_prompt = self._run_with_client(client, run_config, new_request, new_meta, stream)
        self._post_check_output(stream, run_config, new_prompt)
        return new_prompt
    
    @abstractmethod
    async def _arun_with_client(
        self: PromptType, 
        client: OpenAIClient, 
        run_config: RunConfig,
        new_request: dict,
        new_meta: dict,
        stream: bool,
        ) -> PromptType:
        ...
    
    async def arun(
        self: PromptType, 
        client: OpenAIClient = None, 
        run_config: RunConfig = DEFAULT_CONFIG,
        **kwargs) -> PromptType:
        run_config, new_request, new_meta, stream = self._prepare_run(run_config, kwargs)
        if client:
            new_prompt = await self._arun_with_client(client, run_config, new_request, new_meta, stream)
        else:
            async with OpenAIClient(ClientMode.ASYNC) as client:
                new_prompt = await self._arun_with_client(client, run_config, new_request, new_meta, stream)
        self._post_check_output(stream, run_config, new_prompt)
        return new_prompt

    def _prepare_run(self: PromptType, run_config: RunConfig, kwargs: dict):
        new_request = copy.deepcopy(self.request)
        new_request.update(kwargs)
        stream = new_request.get("stream", False)
        new_meta = copy.deepcopy(self.meta)
        # TODO: meta contains origianl run_config; update runtime 
        # run_config according to origianl meta
        
        if run_config.output_evaled_prompt_path \
            or run_config.output_evaled_prompt_fd:
            # output the evaluated prompt to a file or a file descriptor
            evaled_data = self._eval_data(run_config)
            serialized_data = self._serialize_data(evaled_data)
            text = self._dumps(self.request, self.meta, serialized_data)
            if run_config.output_evaled_prompt_path:
                with open(run_config.output_evaled_prompt_path, 'w', encoding='utf-8') as fout:
                    fout.write(text)
            elif run_config.output_evaled_prompt_fd:
                run_config.output_evaled_prompt_fd.write(text)
        return run_config, new_request, new_meta, stream
    
    def _post_check_output(self: PromptType, stream: bool, run_config: RunConfig, new_prompt: PromptType):
        if not stream:
            # if stream is True, the response is already streamed to 
            # a file or a file descriptor
            if run_config.output_path:
                new_prompt.dump_to(run_config.output_path)
            elif run_config.output_fd:
                new_prompt.dump(run_config.output_fd)
        return new_prompt

    def _merge_non_data(self: PromptType, other: PromptType, inplace=False) -> Union[None, tuple[dict, dict]]:
        if inplace:
            merge(self.request, other.request, strategy=Strategy.ADDITIVE)
            merge(self.meta, other.meta, strategy=Strategy.ADDITIVE)
        else:
            merged_request = merge({}, self.request, other.request, strategy=Strategy.ADDITIVE)
            merged_meta = merge({}, self.meta, other.meta, strategy=Strategy.ADDITIVE)
            return merged_request, merged_meta
    
    def _filter_arguments(
        self, request: dict, 
        run_config: RunConfig,
        ) -> dict:
        if run_config.record == RequestRecordMode.BLACKLIST:
            # will modify the original request
            for key in run_config.blacklist:
                request.pop(key, None)
        elif run_config.record == RequestRecordMode.WHITELIST:
            request = {key: value for key, value in request.items() if key in run_config.whitelist}
        elif run_config.record == RequestRecordMode.NONE:
            request = {}
        return request
    
    def _parse_var_map(self, run_config: RunConfig):
        var_map = {}
        if run_config.var_map_path:
            var_map = merge(
                var_map, 
                load_var_map(run_config.var_map_path), 
                strategy=Strategy.REPLACE
            )
        if run_config.var_map:
            var_map = merge(
                var_map, 
                run_config.var_map, 
                strategy=Strategy.REPLACE
            )
        return var_map


class ChatPrompt(HandyPrompt):
        
    def __init__(self, chat: list, request: dict, meta: dict):
        super().__init__(chat, request, meta)
    
    @property
    def chat(self) -> list:
        return self.data
    
    @chat.setter
    def chat(self, value: list):
        self.data = value
    
    @property
    def result_str(self) -> str:
        if len(self.chat) == 0:
            return ""
        return self.chat[-1]['content']
    
    def _serialize_data(self, data) -> str:
        return converter.chat2raw(data)
    
    def _eval_data(self, run_config: RunConfig) -> list:
        var_map = self._parse_var_map(run_config)
        if var_map:
            return converter.chat_replace_variables(
                self.chat, var_map, inplace=False)
        else:
            return self.chat
    
    def _stream_chat_proc(self, request, meta, response, fd: Optional[io.IOBase] = None) -> tuple[str, str]:
        if fd:
            # dump frontmatter
            fd.write(self._dumps(request, meta, "").strip() + "\n\n")
        # stream response to fd
        role = ""
        content = ""
        for r, text in stream_chat_with_role(response):
            if r != role:
                role = r
                if fd:
                    fd.write(f"${role}$\n")
            elif fd:
                fd.write(text)
            content += text
        return role, content
    
    def _run_with_client(
        self, client: OpenAIClient, 
        run_config: RunConfig,
        new_request: dict,
        new_meta: dict,
        stream: bool,
        ) -> ChatPrompt:
        response = client.chat(
            messages=self._eval_data(run_config),
            **new_request
            ).call()
        new_request = self._filter_arguments(new_request, run_config)
        if stream:
            if run_config.output_path:
                # stream response to a file
                with open(run_config.output_path, 'w', encoding='utf-8') as fout:
                    role, content = self._stream_chat_proc(new_request, new_meta, response, fout)
            elif run_config.output_fd:
                # stream response to a file descriptor
                role, content = self._stream_chat_proc(new_request, new_meta, response, run_config.output_fd)
            else:
                role, content = self._stream_chat_proc(new_request, new_meta, response)
        else:
            role = response['choices'][0]['message']['role']
            content = response['choices'][0]['message']['content']
        return ChatPrompt(
            [{"role": role, "content": content}],
            new_request, new_meta
        )
    
    async def _astream_chat_proc(self, request, meta, response, fd: Optional[io.IOBase] = None) -> tuple[str, str]:
        if fd:
            # dump frontmatter
            fd.write(self._dumps(request, meta, "").strip() + "\n\n")
        # stream response to fd
        role = ""
        content = ""
        async for r, text in astream_chat_with_role(response):
            if r != role:
                role = r
                if fd:
                    fd.write(f"${role}$\n")
            elif fd:
                fd.write(text)
            content += text
        return role, content
    
    async def _arun_with_client(
        self, client: OpenAIClient, 
        run_config: RunConfig,
        new_request: dict,
        new_meta: dict,
        stream: bool,
        ) -> ChatPrompt:
        response = await client.chat(
            messages=self._eval_data(run_config),
            **new_request
            ).acall()
        new_request = self._filter_arguments(new_request, run_config)
        if stream:
            if run_config.output_path:
                # stream response to a file
                with open(run_config.output_path, 'w', encoding='utf-8') as fout:
                    role, content = await self._astream_chat_proc(new_request, new_meta, response, fout)
            elif run_config.output_fd:
                # stream response to a file descriptor
                role, content = await self._astream_chat_proc(new_request, new_meta, response, run_config.output_fd)
            else:
                role, content = await self._astream_chat_proc(new_request, new_meta, response)
        else:
            role = response['choices'][0]['message']['role']
            content = response['choices'][0]['message']['content']
        return ChatPrompt(
            [{"role": role, "content": content}],
            new_request, new_meta
        )

    def __add__(self, other: Union[str, list, ChatPrompt]):
        # support concatenation with string, list or another ChatPrompt
        if isinstance(other, str):
            return ChatPrompt(
                self.chat + [{"role": "user", "content": other}],
                copy.deepcopy(self.request),
                copy.deepcopy(self.meta)
            )
        elif isinstance(other, list):
            return ChatPrompt(
                self.chat + [{"role": msg['role'], "content": msg['content']} for msg in other],
                copy.deepcopy(self.request),
                copy.deepcopy(self.meta)
            )
        elif isinstance(other, ChatPrompt):
            # merge two ChatPrompt objects
            merged_request, merged_meta = self._merge_non_data(other)
            return ChatPrompt(
                self.chat + other.chat, merged_request, merged_meta
            )
        else:
            raise TypeError(f"unsupported operand type(s) for +: 'ChatPrompt' and '{type(other)}'")
    
    def __iadd__(self, other: Union[str, list, ChatPrompt]):
        # support concatenation with string, list or another ChatPrompt
        if isinstance(other, str):
            self.chat.append({"role": "user", "content": other})
        elif isinstance(other, list):
            self.chat += [{"role": msg['role'], "content": msg['content']} for msg in other]
        elif isinstance(other, ChatPrompt):
            # merge two ChatPrompt objects
            self.chat += other.chat
            self._merge_non_data(other, inplace=True)
        else:
            raise TypeError(f"unsupported operand type(s) for +: 'ChatPrompt' and '{type(other)}'")
        return self


class CompletionsPrompt(HandyPrompt):
    
    def __init__(self, prompt: str, request: dict, meta: dict):
        super().__init__(prompt, request, meta)
    
    @property
    def prompt(self) -> str:
        return self.data
    
    @prompt.setter
    def prompt(self, value: str):
        self.data = value
    
    def _eval_data(self, run_config: RunConfig) -> str:
        var_map = self._parse_var_map(run_config)
        if var_map:
            new_prompt = self.prompt
            for key, value in var_map.items():
                new_prompt = new_prompt.replace(key, value)
            return new_prompt
        else:
            return self.prompt
    
    def _stream_completions_proc(self, request, meta, response, fd: Optional[io.IOBase] = None) -> str:
        if fd:
            # dump frontmatter
            fd.write(self._dumps(request, meta, "").strip() + "\n\n")
        # stream response to fd
        content = ""
        for text in stream_completions(response):
            if fd:
                fd.write(text)
            content += text
        return content

    def _run_with_client(
        self, client: OpenAIClient, 
        run_config: RunConfig,
        new_request: dict,
        new_meta: dict,
        stream: bool,
        ) -> CompletionsPrompt:
        response = client.completions(
            prompt=self._eval_data(run_config),
            **new_request
            ).call()
        new_request = self._filter_arguments(new_request, run_config)
        if stream:
            if run_config.output_path:
                # stream response to a file
                with open(run_config.output_path, 'w', encoding='utf-8') as fout:
                    content = self._stream_completions_proc(new_request, new_meta, response, fout)
            elif run_config.output_fd:
                # stream response to a file descriptor
                content = self._stream_completions_proc(new_request, new_meta, response, run_config.output_fd)
            else:
                content = self._stream_completions_proc(new_request, new_meta, response)
        else:
            content = response['choices'][0]['text']
        return CompletionsPrompt(content, new_request, new_meta)

    async def _astream_completions_proc(self, request, meta, response, fd: Optional[io.IOBase] = None) -> str:
        if fd:
            # dump frontmatter
            fd.write(self._dumps(request, meta, "").strip() + "\n\n")
        # stream response to fd
        content = ""
        async for text in astream_completions(response):
            if fd:
                fd.write(text)
            content += text
        return content
    
    async def _arun_with_client(
        self, client: OpenAIClient, 
        run_config: RunConfig,
        new_request: dict,
        new_meta: dict,
        stream: bool,
        ) -> CompletionsPrompt:
        response = await client.completions(
            prompt=self._eval_data(run_config),
            **new_request
            ).acall()
        new_request = self._filter_arguments(new_request, run_config)
        if stream:
            if run_config.output_path:
                # stream response to a file
                with open(run_config.output_path, 'w', encoding='utf-8') as fout:
                    content = await self._astream_completions_proc(new_request, new_meta, response, fout)
            elif run_config.output_fd:
                # stream response to a file descriptor
                content = await self._astream_completions_proc(new_request, new_meta, response, run_config.output_fd)
            else:
                content = await self._astream_completions_proc(new_request, new_meta, response)
        else:
            content = response['choices'][0]['text']
        return CompletionsPrompt(content, new_request, new_meta)
    
    def __add__(self, other: Union[str, CompletionsPrompt]):
        # support concatenation with string or another CompletionsPrompt
        if isinstance(other, str):
            return CompletionsPrompt(
                self.prompt + other,
                copy.deepcopy(self.request),
                copy.deepcopy(self.meta)
            )
        elif isinstance(other, CompletionsPrompt):
            # merge two CompletionsPrompt objects
            merged_request, merged_meta = self._merge_non_data(other)
            return CompletionsPrompt(
                self.prompt + other.prompt, merged_request, merged_meta
            )
        else:
            raise TypeError(f"unsupported operand type(s) for +: 'CompletionsPrompt' and '{type(other)}'")
    
    def __iadd__(self, other: Union[str, CompletionsPrompt]):
        # support concatenation with string or another CompletionsPrompt
        if isinstance(other, str):
            self.prompt += other
        elif isinstance(other, CompletionsPrompt):
            # merge two CompletionsPrompt objects
            self.prompt += other.prompt
            self._merge_non_data(other, inplace=True)
        else:
            raise TypeError(f"unsupported operand type(s) for +: 'CompletionsPrompt' and '{type(other)}'")
        return self

