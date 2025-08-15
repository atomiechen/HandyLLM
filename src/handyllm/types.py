__all__ = [
    "PathType",
    "VarMapType",
    "SyncHandlerChat",
    "SyncHandlerCompletions",
    "AsyncHandlerChat",
    "AsyncHandlerCompletions",
    "OnChunkType",
    "StrHandler",
    "StringifyHandler",
    "ShortChatChunk",
]

import sys
from typing import (
    Any,
    Awaitable,
    Callable,
    MutableMapping,
    Optional,
    Tuple,
    TypedDict,
    Union,
)
from os import PathLike

from .response import ToolCallDelta


if sys.version_info >= (3, 9):
    PathType = Union[str, PathLike[str]]
else:
    PathType = Union[str, PathLike]

VarMapType = MutableMapping[str, str]


class ChatChunkDict(TypedDict):
    role: str
    content: Optional[str]
    reasoning_content: Optional[str]
    tool_call: ToolCallDelta


SyncHandlerChat = Callable[[ChatChunkDict], Any]
SyncHandlerCompletions = Callable[[str], Any]
AsyncHandlerChat = Callable[[ChatChunkDict], Awaitable[Any]]
AsyncHandlerCompletions = Callable[[str], Awaitable[Any]]
OnChunkType = Union[
    SyncHandlerChat, SyncHandlerCompletions, AsyncHandlerChat, AsyncHandlerCompletions
]

StrHandler = Callable[[str], Any]
StringifyHandler = Callable[[Any], str]

ShortChatChunk = Tuple[str, Optional[str], Optional[str], ToolCallDelta]
