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

from enum import Enum, auto
from abc import abstractmethod, ABC
import io
from typing import Optional, Union, TypeVar
import re
import copy
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
    path: str,
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
    path: str
) -> None:
    return prompt.dump_to(path)

def load_var_map(path: str) -> dict[str, str]:
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
    record: RequestRecordMode = RequestRecordMode.NONE
    blacklist: Optional[list[str]] = None
    whitelist: Optional[list[str]] = None
    var_map: Optional[dict[str, str]] = None
    var_map_path: Optional[str] = None


DEFAULT_CONFIG = RunConfig(
    record=RequestRecordMode.BLACKLIST,
    blacklist=DEFAULT_BLACKLIST,
    whitelist=None,
    var_map=None,
    var_map_path=None,
)


class HandyPrompt(ABC):
    
    def __init__(self, data: Union[str, list], request: dict = None, meta: dict = None):
        self.data = data
        self.request = request or {}
        self.meta = meta or {}
    
    @property
    def result_str(self) -> str:
        return str(self.data)
    
    def _serialize_data(self) -> str:
        '''
        Serialize the data to a string. 
        This method can be overridden by subclasses.
        '''
        return str(self.data)
    
    def dumps(self) -> str:
        serialized_data = self._serialize_data()
        if not self.meta and not self.request:
            return serialized_data
        else:
            front_data = copy.deepcopy(self.request)
            if self.meta:
                front_data['meta'] = copy.deepcopy(self.meta)
            post = frontmatter.Post(serialized_data, None, **front_data)
            return frontmatter.dumps(post, handler)
    
    def dump(self, fd: io.IOBase) -> None:
        text = self.dumps()
        fd.write(text)
    
    def dump_to(self, path: str) -> None:
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
        **kwargs) -> PromptType:
        ...
    
    def run(
        self: PromptType, 
        client: OpenAIClient = None, 
        run_config: RunConfig = DEFAULT_CONFIG,
        **kwargs) -> PromptType:
        if client:
            return self._run_with_client(client, run_config, **kwargs)
        else:
            with OpenAIClient(ClientMode.SYNC) as client:
                return self._run_with_client(client, run_config, **kwargs)
    
    @abstractmethod
    async def _arun_with_client(
        self: PromptType, 
        client: OpenAIClient, 
        run_config: RunConfig,
        **kwargs) -> PromptType:
        ...
    
    async def arun(
        self: PromptType, 
        client: OpenAIClient = None, 
        run_config: RunConfig = DEFAULT_CONFIG,
        **kwargs) -> PromptType:
        if client:
            return await self._arun_with_client(client, run_config, **kwargs)
        else:
            async with OpenAIClient(ClientMode.ASYNC) as client:
                return await self._arun_with_client(client, run_config, **kwargs)

    def _merge_non_data(self: PromptType, other: PromptType, inplace=False) -> Union[None, tuple[dict, dict]]:
        if inplace:
            merge(self.request, other.request, strategy=Strategy.ADDITIVE)
            merge(self.meta, other.meta, strategy=Strategy.ADDITIVE)
        else:
            merged_request = merge({}, self.request, other.request, strategy=Strategy.ADDITIVE)
            merged_meta = merge({}, self.meta, other.meta, strategy=Strategy.ADDITIVE)
            return merged_request, merged_meta
    
    def _new_arguments(
        self, arguments: dict, record: RequestRecordMode, 
        blacklist: list[str], whitelist: list[str]) -> dict:
        if record == RequestRecordMode.BLACKLIST:
            # will modify the original arguments
            for key in blacklist:
                arguments.pop(key, None)
        elif record == RequestRecordMode.WHITELIST:
            arguments = {key: value for key, value in arguments.items() if key in whitelist}
        elif record == RequestRecordMode.NONE:
            arguments = {}
        return arguments
    
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
    
    def _serialize_data(self) -> str:
        return converter.chat2raw(self.chat)
    
    def _eval_data(self, run_config: RunConfig) -> list:
        var_map = self._parse_var_map(run_config)
        if var_map:
            return converter.chat_replace_variables(
                self.chat, var_map, inplace=False)
        else:
            return self.chat
    
    def _run_with_client(
        self, client: OpenAIClient, 
        run_config: RunConfig,
        **kwargs) -> ChatPrompt:
        arguments = copy.deepcopy(self.request)
        arguments.update(kwargs)
        stream = arguments.get("stream", False)
        response = client.chat(
            messages=self._eval_data(run_config),
            **arguments
            ).call()
        if stream:
            role = ""
            content = ""
            for r, text in stream_chat_with_role(response):
                role = r
                content += text
        else:
            role = response['choices'][0]['message']['role']
            content = response['choices'][0]['message']['content']
        return ChatPrompt(
            [{"role": role, "content": content}],
            self._new_arguments(arguments, run_config.record, run_config.blacklist, run_config.whitelist),
            copy.deepcopy(self.meta)
        )
    
    async def _arun_with_client(
        self, client: OpenAIClient, 
        run_config: RunConfig,
        **kwargs) -> ChatPrompt:
        arguments = copy.deepcopy(self.request)
        arguments.update(kwargs)
        stream = arguments.get("stream", False)
        response = await client.chat(
            messages=self._eval_data(run_config),
            **arguments
            ).acall()
        if stream:
            role = ""
            content = ""
            async for r, text in astream_chat_with_role(response):
                role = r
                content += text
        else:
            role = response['choices'][0]['message']['role']
            content = response['choices'][0]['message']['content']
        return ChatPrompt(
            [{"role": role, "content": content}],
            self._new_arguments(arguments, run_config.record, run_config.blacklist, run_config.whitelist),
            copy.deepcopy(self.meta)
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
        else:
            return self.prompt

    def _run_with_client(
        self, client: OpenAIClient, 
        run_config: RunConfig,
        **kwargs) -> CompletionsPrompt:
        arguments = copy.deepcopy(self.request)
        arguments.update(kwargs)
        stream = arguments.get("stream", False)
        response = client.completions(
            prompt=self._eval_data(run_config),
            **arguments
            ).call()
        if stream:
            content = ""
            for text in stream_completions(response):
                content += text
        else:
            content = response['choices'][0]['text']
        return CompletionsPrompt(
            content,
            self._new_arguments(arguments, run_config.record, run_config.blacklist, run_config.whitelist),
            copy.deepcopy(self.meta)
        )
    
    async def _arun_with_client(
        self, client: OpenAIClient, 
        run_config: RunConfig,
        **kwargs) -> CompletionsPrompt:
        arguments = copy.deepcopy(self.request)
        arguments.update(kwargs)
        stream = arguments.get("stream", False)
        response = await client.completions(
            prompt=self._eval_data(run_config),
            **arguments
            ).acall()
        if stream:
            content = ""
            async for text in astream_completions(response):
                content += text
        else:
            content = response['choices'][0]['text']
        return CompletionsPrompt(
            content,
            self._new_arguments(arguments, run_config.record, run_config.blacklist, run_config.whitelist),
            copy.deepcopy(self.meta)
        )
    
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

