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

from abc import abstractmethod, ABC
import io
from typing import Union
import copy

import frontmatter
from mergedeep import merge, Strategy

from .prompt_converter import PromptConverter
from .openai_client import OpenAIClient
from .utils import stream_chat_with_role, stream_completions


converter = PromptConverter()
handler = frontmatter.YAMLHandler()


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
    def _run_with_client(self, client: OpenAIClient) -> HandyPrompt:
        ...
    
    def run(self, client: OpenAIClient = None) -> HandyPrompt:
        if client:
            return self._run_with_client(client)
        else:
            with OpenAIClient() as client:
                return self._run_with_client(client)


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
    
    def _run_with_client(self, client: OpenAIClient) -> ChatPrompt:
        arguments = copy.deepcopy(self.request)
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
            arguments,
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
            return ChatPrompt(
                self.chat + other.chat,
                merge({}, self.request, other.request, strategy=Strategy.ADDITIVE),
                merge({}, self.meta, other.meta, strategy=Strategy.ADDITIVE)
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
            merge(self.request, other.request, strategy=Strategy.ADDITIVE)
            merge(self.meta, other.meta, strategy=Strategy.ADDITIVE)
        else:
            raise TypeError(f"unsupported operand type(s) for +: 'ChatPrompt' and '{type(other)}'")
        return self


class CompletionsPrompt(HandyPrompt):
    
    def __init__(self, prompt: str, request: dict, meta: dict):
        super().__init__(prompt, request, meta)
    
    @property
    def prompt(self) -> str:
        return self.data
    
    def _serialize_data(self) -> str:
        return self.prompt

    def _run_with_client(self, client: OpenAIClient) -> CompletionsPrompt:
        arguments = copy.deepcopy(self.request)
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
            arguments,
            copy.deepcopy(self.meta)
        )

