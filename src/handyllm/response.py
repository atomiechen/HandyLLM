__all__ = [
    "Message",
    "ChatChoice",
    "ChatResponse",
    "ChatChunkChoice",
    "ChatChunk",
    "CompletionChoice",
    "CompletionsResponse",
    "CompletionsChunkChoice",
    "CompletionsChunk",
    "Function",
    "ToolCall",
    "TopLogProbItem",
    "LogProbItem",
    "Logprobs",
    "Usage",
    "ToolCallDelta",
    "ChatChunkDelta",
    "CompletionLogprobs",
]

from typing import List, Optional, TypedDict
from typing_extensions import NotRequired


class Function(TypedDict):
    name: str
    arguments: str


class ToolCall(TypedDict):
    id: str
    type: str
    function: Function


class Message(TypedDict):
    role: str
    content: Optional[str]
    tool_calls: NotRequired[List[ToolCall]]


class TopLogProbItem(TypedDict):
    token: str
    logprob: float
    bytes: Optional[List[int]]


class LogProbItem(TypedDict):
    token: str
    logprob: float
    bytes: Optional[List[int]]
    top_logprobs: List[TopLogProbItem]


class Logprobs(TypedDict):
    content: Optional[List[LogProbItem]]


class ChatChoice(TypedDict):
    index: int
    message: Message
    logprob: Optional[Logprobs]
    finish_reason: str


class Usage(TypedDict):
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int


class ChatResponse(TypedDict):
    id: str
    choices: List[ChatChoice]
    created: int
    model: str
    service_tier: Optional[str]
    system_fingerprint: str
    object: str
    usage: Usage


class ToolCallDelta(TypedDict):
    index: int
    id: str
    type: str
    function: Function


class ChatChunkDelta(TypedDict):
    """
    Note that following keys may not exist in the delta dict.
    """

    role: NotRequired[str]
    content: NotRequired[Optional[str]]
    tool_calls: NotRequired[List[ToolCallDelta]]


class ChatChunkChoice(TypedDict):
    delta: ChatChunkDelta
    logprobs: Optional[Logprobs]
    finish_reason: Optional[str]
    index: int


class ChatChunk(TypedDict):
    id: str
    choices: List[ChatChunkChoice]
    created: int
    model: str
    service_tier: Optional[str]
    system_fingerprint: str
    object: str
    usage: Optional[Usage]


class CompletionLogprobs(TypedDict):
    text_offset: list
    token_logprobs: list
    tokens: list
    top_logprobs: list


class CompletionChoice(TypedDict):
    finish_reason: str
    index: int
    logprobs: Optional[CompletionLogprobs]
    text: str


class CompletionsResponse(TypedDict):
    id: str
    choices: List[CompletionChoice]
    created: int
    model: str
    system_fingerprint: str
    object: str
    usage: Usage


class CompletionsChunkChoice(TypedDict):
    text: str


class CompletionsChunk(TypedDict):
    choices: List[CompletionsChunkChoice]
