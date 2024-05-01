from __future__ import annotations
__all__ = [
    "HandyPrompt",
    "ChatPrompt",
    "CompletionsPrompt",
    "loads",
    "load",
    "dumps",
    "dump",
]

from abc import abstractmethod, ABC
import io
from typing import Union
import copy

import frontmatter

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
        meta, content = frontmatter.parse(text, encoding, handler)
    else:
        meta = {}
        content = text
    api: str = meta.get("api", "")
    if api.startswith("completion"):
        return CompletionsPrompt(content, meta)
    else:
        chat = converter.raw2chat(content)
        return ChatPrompt(chat, meta)

def load(
    fd: io.IOBase, 
    encoding: str = "utf-8"
) -> HandyPrompt:
    text = fd.read()
    return loads(text, encoding)

def dumps(
    prompt: HandyPrompt, 
) -> str:
    return prompt.dumps()

def dump(
    prompt: HandyPrompt, 
    fd: io.IOBase, 
) -> None:
    return prompt.dump(fd)


class HandyPrompt(ABC):
    
    def __init__(self, data: Union[str, list], meta: dict = None):
        self.data = data
        self.meta = meta or {}
    
    @abstractmethod
    def _serialize_data(self) -> str:
        '''
        Serialize the data to a string. 
        This method should be implemented by subclasses.
        '''
    
    def dumps(self) -> str:
        serialized_data = self._serialize_data()
        if not self.meta:
            return serialized_data
        else:
            post = frontmatter.Post(serialized_data, None, **self.meta)
            return frontmatter.dumps(post, handler)
    
    def dump(self, fd: io.IOBase) -> None:
        text = self.dumps()
        fd.write(text)
    
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
        
    def __init__(self, chat: list, meta: dict):
        super().__init__(chat, meta)
    
    @property
    def chat(self) -> list:
        return self.data
    
    def _serialize_data(self) -> str:
        return converter.chat2raw(self.chat)
    
    def _run_with_client(self, client: OpenAIClient) -> ChatPrompt:
        arguments = copy.deepcopy(self.meta)
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
            arguments
        )


class CompletionsPrompt(HandyPrompt):
    
    def __init__(self, prompt: str, meta: dict):
        super().__init__(prompt, meta)
    
    @property
    def prompt(self) -> str:
        return self.data
    
    def _serialize_data(self) -> str:
        return self.prompt

    def _run_with_client(self, client: OpenAIClient) -> CompletionsPrompt:
        arguments = copy.deepcopy(self.meta)
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
            arguments
        )

