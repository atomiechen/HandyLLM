import sys
from typing import Any, Awaitable, Callable, Dict, MutableMapping, Optional, Union
from os import PathLike


if sys.version_info >= (3, 9):
    PathType = Union[str, PathLike[str]]
else:
    PathType = Union[str, PathLike]

VarMapType = MutableMapping[str, str]

SyncHandlerChat = Callable[[str, Optional[str], Optional[Dict]], Any]
SyncHandlerCompletions = Callable[[str], Any]
AsyncHandlerChat = Callable[[str, Optional[str], Optional[Dict]], Awaitable[Any]]
AsyncHandlerCompletions = Callable[[str], Awaitable[Any]]
OnChunkType = Union[SyncHandlerChat, SyncHandlerCompletions, AsyncHandlerChat, AsyncHandlerCompletions]

