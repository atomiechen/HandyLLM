__all__ = [
    'DictProxy',
    'Message',
    'ChatChoice',
    'ChatResponse',
    'ChatChunkChoice',
    'ChatChunk',
    'CompletionChoice',
    'CompletionsResponse',
    'CompletionsChunkChoice',
    'CompletionsChunk',    
]

from typing import MutableMapping, Optional, Sequence, TypedDict
from typing_extensions import NotRequired


class DictProxy(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            self[key] = self._wrap(value)

    def __getattr__(self, attr):
        return self[attr]

    def __setattr__(self, attr, value):
        self[attr] = self._wrap(value)
    
    def _wrap(self, value):
        if isinstance(value, DictProxy):
            return value
        elif isinstance(value, MutableMapping):
            return DictProxy(value)
        elif isinstance(value, Sequence) and not isinstance(value, str):
            return [self._wrap(v) for v in value]
        return value


class Function(DictProxy):
    name: str
    arguments: str


class ToolCall(DictProxy):
    id: str
    type: str
    function: Function


class Message(TypedDict):
    role: str
    content: Optional[str]
    tool_calls: NotRequired[list[ToolCall]]


class TopLogProbItem(DictProxy):
    token: str
    logprob: float
    bytes: Optional[list[int]]


class LogProbItem(DictProxy):
    token: str
    logprob: float
    bytes: Optional[list[int]]
    top_logprobs: list[TopLogProbItem]


class Logprobs(DictProxy):
    content: Optional[list[LogProbItem]]


class ChatChoice(DictProxy):
    index: int
    message: Message
    logprob: Optional[Logprobs]
    finish_reason: str


class Usage(DictProxy):
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int


class ChatResponse(DictProxy):
    id: str
    choices: list[ChatChoice]
    created: int
    model: str
    service_tier: Optional[str]
    system_fingerprint: str
    object: str
    usage: Usage


class ToolCallDelta(DictProxy):
    index: int
    id: str
    type: str
    function: Function


class ChatChunkDelta(TypedDict):
    '''
    Note that following keys may not exist in the delta dict.
    '''
    role: NotRequired[str]
    content: NotRequired[Optional[str]]
    tool_calls: NotRequired[list[ToolCallDelta]]


class ChatChunkChoice(DictProxy):
    delta: ChatChunkDelta
    logprobs: Optional[Logprobs]
    finish_reason: Optional[str]
    index: int


class ChatChunk(DictProxy):
    id: str
    choices: list[ChatChunkChoice]
    created: int
    model: str
    service_tier: Optional[str]
    system_fingerprint: str
    object: str
    usage: Optional[Usage]


class CompletionLogprobs(DictProxy):
    text_offset: list
    token_logprobs: list
    tokens: list
    top_logprobs: list


class CompletionChoice(DictProxy):
    finish_reason: str
    index: int
    logprobs: Optional[CompletionLogprobs]
    text: str


class CompletionsResponse(DictProxy):
    id: str
    choices: list[CompletionChoice]
    created: int
    model: str
    system_fingerprint: str
    object: str
    usage: Usage


class CompletionsChunkChoice(DictProxy):
    text: str


class CompletionsChunk(DictProxy):
    choices: list[CompletionsChunkChoice]

