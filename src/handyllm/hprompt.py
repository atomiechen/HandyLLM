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
]

from enum import Enum, auto
from abc import abstractmethod, ABC
import io
from typing import Union, TypeVar
import copy

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

DEFAULT_BLACKLIST = [
    "api_key", "organization", "api_base", "api_type", "api_version", 
    "endpoint_manager", "endpoint", "engine", "deployment_id", 
    "model_engine_map", "dest_url", 
]


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


class RequestRecordMode(Enum):
    BLACKLIST = auto()  # record all request arguments except specified ones
    WHITELIST = auto()  # record only specified request arguments
    NONE = auto()  # record no request arguments
    ALL = auto()  # record all request arguments


class HandyPrompt(ABC):
    
    def __init__(self, data: Union[str, list], request: dict = None, meta: dict = None):
        self.data = data
        self.request = request or {}
        self.meta = meta or {}
    
    @abstractmethod
    def _serialize_data(self) -> str:
        '''
        Serialize the data to a string. 
        This method should be implemented by subclasses.
        '''
    
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
    def _run_with_client(
        self: PromptType, 
        client: OpenAIClient, 
        record: RequestRecordMode,
        blacklist: list[str],
        whitelist: list[str],
        **kwargs) -> PromptType:
        ...
    
    def run(
        self: PromptType, 
        client: OpenAIClient = None, 
        record: RequestRecordMode = RequestRecordMode.BLACKLIST,
        blacklist: list[str] = DEFAULT_BLACKLIST,
        whitelist: list[str] = None,
        **kwargs) -> PromptType:
        if client:
            return self._run_with_client(
                client=client,
                record=record,
                blacklist=blacklist,
                whitelist=whitelist,
                **kwargs)
        else:
            with OpenAIClient(ClientMode.SYNC) as client:
                return self._run_with_client(
                    client=client,
                    record=record,
                    blacklist=blacklist,
                    whitelist=whitelist,
                    **kwargs)
    
    @abstractmethod
    async def _arun_with_client(
        self: PromptType, 
        client: OpenAIClient, 
        record: RequestRecordMode,
        blacklist: list[str],
        whitelist: list[str],
        **kwargs) -> PromptType:
        ...
    
    async def arun(
        self: PromptType, 
        client: OpenAIClient = None, 
        record: RequestRecordMode = RequestRecordMode.BLACKLIST,
        blacklist: list[str] = DEFAULT_BLACKLIST,
        whitelist: list[str] = None,
        **kwargs) -> PromptType:
        if client:
            return await self._arun_with_client(
                client=client,
                record=record,
                blacklist=blacklist,
                whitelist=whitelist,
                **kwargs)
        else:
            async with OpenAIClient(ClientMode.ASYNC) as client:
                return await self._arun_with_client(
                    client=client,
                    record=record,
                    blacklist=blacklist,
                    whitelist=whitelist,
                    **kwargs)

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


class ChatPrompt(HandyPrompt):
        
    def __init__(self, chat: list, request: dict, meta: dict):
        super().__init__(chat, request, meta)
    
    @property
    def chat(self) -> list:
        return self.data
    
    @chat.setter
    def chat(self, value: list):
        self.data = value
    
    def _serialize_data(self) -> str:
        return converter.chat2raw(self.chat)
    
    def _run_with_client(
        self, client: OpenAIClient, 
        record: RequestRecordMode, 
        blacklist: list[str],
        whitelist: list[str],
        **kwargs) -> ChatPrompt:
        arguments = copy.deepcopy(self.request)
        arguments.update(kwargs)
        stream = arguments.get("stream", False)
        response = client.chat(
            messages=self.chat,
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
            self._new_arguments(arguments, record, blacklist, whitelist),
            copy.deepcopy(self.meta)
        )
    
    async def _arun_with_client(
        self, client: OpenAIClient, 
        record: RequestRecordMode,
        blacklist: list[str],
        whitelist: list[str],
        **kwargs) -> ChatPrompt:
        arguments = copy.deepcopy(self.request)
        arguments.update(kwargs)
        stream = arguments.get("stream", False)
        response = await client.chat(
            messages=self.chat,
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
            self._new_arguments(arguments, record, blacklist, whitelist),
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
    
    def _serialize_data(self) -> str:
        return self.prompt

    def _run_with_client(
        self, client: OpenAIClient, 
        record: RequestRecordMode, 
        blacklist: list[str],
        whitelist: list[str],
        **kwargs) -> CompletionsPrompt:
        arguments = copy.deepcopy(self.request)
        arguments.update(kwargs)
        stream = arguments.get("stream", False)
        response = client.completions(
            prompt=self.prompt,
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
            self._new_arguments(arguments, record, blacklist, whitelist),
            copy.deepcopy(self.meta)
        )
    
    async def _arun_with_client(
        self, client: OpenAIClient, 
        record: RequestRecordMode,
        blacklist: list[str],
        whitelist: list[str],
        **kwargs) -> CompletionsPrompt:
        arguments = copy.deepcopy(self.request)
        arguments.update(kwargs)
        stream = arguments.get("stream", False)
        response = await client.completions(
            prompt=self.prompt,
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
            self._new_arguments(arguments, record, blacklist, whitelist),
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

