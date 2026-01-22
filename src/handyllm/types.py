import sys
from typing import (
    Any,
    Awaitable,
    Callable,
    List,
    Literal,
    MutableMapping,
    Optional,
    Tuple,
    TypedDict,
    Union,
)
from os import PathLike

from typing_extensions import NotRequired


if sys.version_info >= (3, 9):
    PathType = Union[str, PathLike[str]]
else:
    PathType = Union[str, PathLike]

VarMapType = MutableMapping[str, str]


class Function(TypedDict):
    name: str
    arguments: str


class FunctionToolCall(TypedDict):
    id: str
    type: Literal["function"]
    function: Function


class CustomFunction(TypedDict):
    input: str
    name: str


class CustomToolCall(TypedDict):
    id: str
    type: Literal["custom"]
    custom: CustomFunction


ToolCall = Union[FunctionToolCall, CustomToolCall]


class ResponseMessage(TypedDict):
    role: str
    content: Optional[str]
    reasoning_content: NotRequired[Optional[str]]
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
    message: ResponseMessage
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
    reasoning_content: NotRequired[Optional[str]]
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
    text_offset: List
    token_logprobs: List
    tokens: List
    top_logprobs: List


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


class TextContentPart(TypedDict):
    type: Literal["text"]
    text: str


class ImageUrl(TypedDict):
    url: str
    detail: NotRequired[str]


class ImageContentPart(TypedDict):
    type: Literal["image_url"]
    image_url: ImageUrl


class InputAudio(TypedDict):
    data: str
    format: str


class AudioContentPart(TypedDict):
    type: Literal["input_audio"]
    input_audio: InputAudio


class FileObject(TypedDict):
    file_data: NotRequired[str]
    file_id: NotRequired[str]
    filename: NotRequired[str]


class FileContentPart(TypedDict):
    type: Literal["file"]
    file: FileObject


class DeveloperMessage(TypedDict):
    role: Literal["developer"]
    content: Union[str, List[TextContentPart]]
    name: NotRequired[str]


class SystemMessage(TypedDict):
    role: Literal["system"]
    content: Union[str, List[TextContentPart]]
    name: NotRequired[str]


class UserMessage(TypedDict):
    role: Literal["user"]
    content: Union[
        str,
        List[
            Union[TextContentPart, ImageContentPart, AudioContentPart, FileContentPart]
        ],
    ]
    name: NotRequired[str]


class RefusalContentPart(TypedDict):
    type: Literal["refusal"]
    refusal: str


class AssistantMessage(TypedDict):
    role: Literal["assistant"]
    content: Union[str, List[Union[TextContentPart, RefusalContentPart]]]
    name: NotRequired[str]
    refusal: NotRequired[str]
    tool_calls: NotRequired[List[ToolCall]]


class ToolMessage(TypedDict):
    role: Literal["tool"]
    content: str
    tool_call_id: str


InputMessage = Union[
    DeveloperMessage, SystemMessage, UserMessage, AssistantMessage, ToolMessage
]
Message = Union[InputMessage, ResponseMessage]


class ChatChunkUnified(TypedDict):
    role: str
    content: Optional[str]
    reasoning_content: Optional[str]
    tool_call: ToolCallDelta


SyncHandlerChat = Callable[[ChatChunkUnified], Any]
SyncHandlerCompletions = Callable[[str], Any]
AsyncHandlerChat = Callable[[ChatChunkUnified], Awaitable[Any]]
AsyncHandlerCompletions = Callable[[str], Awaitable[Any]]
OnChunkType = Union[
    SyncHandlerChat, SyncHandlerCompletions, AsyncHandlerChat, AsyncHandlerCompletions
]

StrHandler = Callable[[str], Any]
StringifyHandler = Callable[[Any], str]

ShortChatChunk = Tuple[str, Optional[str], Optional[str], ToolCallDelta]
