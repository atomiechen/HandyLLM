from __future__ import annotations

import inspect
import re
import copy
import io
import sys
from pathlib import Path
from datetime import datetime
from typing import (
    IO,
    AsyncGenerator,
    Dict,
    Generator,
    Generic,
    Iterable,
    List,
    Literal,
    MutableMapping,
    Optional,
    Tuple,
    Type,
    Union,
    TypeVar,
    cast,
)
from abc import abstractmethod, ABC
from contextlib import asynccontextmanager, contextmanager

import frontmatter
from mergedeep import merge as merge_dict, Strategy
from dotenv import load_dotenv

from .prompt_converter import PromptConverter
from .openai_client import ClientMode, OpenAIClient
from .requestor import Requestor
from .utils import (
    astream_chat_all,
    astream_completions,
    content_part_audio,
    content_part_file,
    content_part_image,
    content_part_text,
    echo_consumer,
    trans_stream_chat,
    file_uri_to_base64,
    file_uri_to_base64_image,
    stream_chat_all,
    stream_completions,
)
from .run_config import RunConfig, RecordRequestMode, CredentialType, VarMapFileFormat
from .types import (
    AudioContentPart,
    FileContentPart,
    ImageContentPart,
    InputMessage,
    PathType,
    RefusalContentPart,
    ResponseMessage,
    SyncHandlerChat,
    SyncHandlerCompletions,
    TextContentPart,
    ToolCall,
    VarMapType,
    ChatChunk,
    ChatResponse,
    CompletionsChunk,
    CompletionsResponse,
    Message,
)
from ._io import MySafeDumper, json_load, yaml_load


PromptType = TypeVar("PromptType", bound="HandyPrompt")
ResponseType = TypeVar("ResponseType")
YieldType = TypeVar("YieldType")
DataType = TypeVar("DataType")


converter = PromptConverter()
handler = frontmatter.YAMLHandler()


p_var_map = re.compile(r"(%\w+%)")

DEFAULT_CONFIG = RunConfig()
DEFAULT_BLACKLIST = (
    "api_key",
    "organization",
    "api_base",
    "api_type",
    "api_version",
    "endpoint_manager",
    "endpoint",
    "engine",
    "deployment_id",
    "model_engine_map",
    "dest_url",
    "endpoints",
)


class HandyPrompt(ABC, Generic[ResponseType, YieldType, DataType]):
    TEMPLATE_TIMESTAMP = "%Y%m%d_%H%M%S"
    TEMPLATE_OUTPUT_FILENAME = f"result_{TEMPLATE_TIMESTAMP}.hprompt"
    TEMPLATE_OUTPUT_EVAL_FILENAME = f"evaled_{TEMPLATE_TIMESTAMP}.hprompt"

    def __init__(
        self,
        data: DataType,
        request: Optional[MutableMapping] = None,
        meta: Optional[Union[MutableMapping, RunConfig]] = None,
        base_path: Optional[PathType] = None,
        response: Optional[ResponseType] = None,
        requestor: Optional[Requestor] = None,
    ):
        self.data: DataType = data
        self.request = request or {}
        # parse meta to run_config
        if isinstance(meta, RunConfig):
            self.run_config = meta
        else:
            self.run_config = RunConfig.from_dict(meta or {}, base_path=base_path)
        self.base_path = base_path
        self.response = response
        self.requestor = requestor

    def __str__(self) -> str:
        return str(self.data)

    def __repr__(self) -> str:
        return "{}({}, {}, {})".format(
            type(self).__name__,
            repr(self.data),
            repr(self.request),
            repr(self.run_config),
        )

    @property
    def result_str(self) -> str:
        """
        Get the result string from the data.
        """
        return str(self.data)

    @classmethod
    def loads(
        cls: Type[PromptType],
        text: str,
        encoding: str = "utf-8",
        base_path: Optional[PathType] = None,
    ) -> PromptType:
        """
        Load a HandyPrompt from a string.

        The returned prompt type depends on the following rules:
        - If called from `HandyPrompt`, the prompt type is *auto* detected.
        - If called from a subclass, the *specified* subclass is used.
        """
        if handler.detect(text):
            metadata, data = frontmatter.parse(text, encoding, handler)
            meta = metadata.pop("meta", None)
            if not isinstance(meta, dict):
                meta = {}
            request = metadata
        else:
            data = text
            request = {}
            meta = {}
        if cls == HandyPrompt:
            # get specific prompt class
            api: str = meta.get("api", "")
            if api:
                api = api.lower()
                if api.startswith("chat"):
                    cls = cast(Type[PromptType], ChatPrompt)
                else:
                    cls = cast(Type[PromptType], CompletionsPrompt)
            else:
                if converter.detect(data):
                    cls = cast(Type[PromptType], ChatPrompt)
                else:
                    cls = cast(Type[PromptType], CompletionsPrompt)
        if cls == ChatPrompt:
            data = converter.raw2msgs(data)
        return cls(data, request, meta, base_path)

    @classmethod
    def load(
        cls,
        fd: IO[str],
        encoding: str = "utf-8",
        base_path: Optional[PathType] = None,
    ):
        """
        Load a HandyPrompt from an `IO[str]` (e.g., file descriptor) (must support `read()`).

        The returned prompt type depends on the following rules:
        - If called from `HandyPrompt`, the prompt type is *auto* detected.
        - If called from a subclass, the *specified* subclass is used.
        """
        text = fd.read()
        return cls.loads(text, encoding, base_path=base_path)

    @classmethod
    def load_from(
        cls,
        path: PathType,
        encoding: str = "utf-8",
    ):
        """
        Load a HandyPrompt from a path.

        The returned prompt type depends on the following rules:
        - If called from `HandyPrompt`, the prompt type is *auto* detected.
        - If called from a subclass, the *specified* subclass is used.
        """
        with open(path, "r", encoding=encoding) as fd:
            return cls.load(fd, encoding, base_path=Path(path).parent.resolve())

    @staticmethod
    def _serialize_data(data: DataType) -> str:
        """
        Serialize the data to a string.
        This method can be overridden by subclasses.
        """
        return str(data)

    @classmethod
    def _dumps_frontmatter(
        cls,
        request: MutableMapping,
        run_config: RunConfig,
        base_path: Optional[PathType] = None,
    ) -> str:
        # dump frontmatter
        if not run_config and not request:
            return ""
        # need to filter the request
        front_data = cls._filter_request(request, run_config)
        if run_config:
            front_data["meta"] = run_config.to_dict(
                retain_object=False, base_path=base_path
            )
        post = frontmatter.Post("", None, **front_data)
        return frontmatter.dumps(post, handler, Dumper=MySafeDumper).strip() + "\n\n"

    @classmethod
    def _dump_fd_if_set(cls, run_config: RunConfig, request: MutableMapping, data):
        with cls.open_and_dump_frontmatter(run_config, request) as fout:
            if fout:
                fout.write(cls._serialize_data(data))

    def dumps(self, base_path: Optional[PathType] = None) -> str:
        """
        Dump this HandyPrompt to a string.
        """
        serialized_data = self._serialize_data(self.data)
        base_path = base_path or self.base_path
        return (
            type(self)._dumps_frontmatter(self.request, self.run_config, base_path)
            + serialized_data
        )

    def dump(self, fd: IO[str], base_path: Optional[PathType] = None) -> None:
        """
        Dump this HandyPrompt to an `IO[str]` (e.g., file descriptor) (must support `write()`).
        """
        text = self.dumps(base_path=base_path)
        fd.write(text)

    def dump_to(self, path: PathType, mkdir: bool = False) -> None:
        """
        Dump this HandyPrompt to a file.
        """
        if mkdir:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fd:
            self.dump(fd, base_path=Path(path).parent.resolve())

    @abstractmethod
    def _eval_data(self, var_map: MutableMapping) -> DataType: ...

    def eval(
        self: PromptType,
        run_config: RunConfig = DEFAULT_CONFIG,
        var_map: Optional[VarMapType] = None,
        **kwargs,
    ) -> PromptType:
        """
        Evaluate the prompt with the given run_config. var_map is for convenience.
        A new prompt object is returned.
        """
        new_run_config = self.eval_run_config(run_config)
        if var_map:
            if new_run_config.var_map is None:
                new_run_config.var_map = {}
            # merge var_map instead of replacing as a whole
            merge_dict(new_run_config.var_map, var_map, strategy=Strategy.REPLACE)
        var_map = self._parse_var_map(new_run_config)
        new_data = self._eval_data(var_map)
        # remove the var_map related config from the new run_config, as it is already applied
        new_run_config.var_map = None
        new_run_config.var_map_path = None
        new_run_config.var_map_file_format = None
        # update the request with the keyword arguments
        evaled_request = copy.deepcopy(self.request)
        evaled_request.update(kwargs)
        return type(self)(
            new_data,
            evaled_request,
            new_run_config,
        )

    def eval_run_config(
        self,
        run_config: RunConfig,
    ) -> RunConfig:
        # merge runtime run_config with the original run_config
        run_config = self.run_config.merge(run_config)

        cls = type(self)
        start_time = datetime.now()
        if run_config.output_path:
            run_config.output_path = cls._prepare_output_path(
                run_config.output_path,
                start_time,
                cls.TEMPLATE_OUTPUT_FILENAME,
                run_config.overwrite_if_exists or False,
            )
        if run_config.output_evaled_prompt_path:
            run_config.output_evaled_prompt_path = cls._prepare_output_path(
                run_config.output_evaled_prompt_path,
                start_time,
                cls.TEMPLATE_OUTPUT_EVAL_FILENAME,
                run_config.overwrite_if_exists or False,
            )

        if run_config.credential_path:
            if not run_config.credential_type:
                # guess the credential type from the file extension
                suffix = Path(run_config.credential_path).suffix[1:]
                if suffix == "yml":
                    suffix = "yaml"
                if suffix in CredentialType:
                    run_config.credential_type = suffix  # type: ignore
                else:
                    run_config.credential_type = CredentialType.ENV

        if run_config.var_map_path:
            if not run_config.var_map_file_format:
                # guess the var_map file format from the file extension
                suffix = Path(run_config.var_map_path).suffix[1:]
                if suffix == "yml":
                    suffix = "yaml"
                if suffix in VarMapFileFormat:
                    run_config.var_map_file_format = suffix  # type: ignore
                else:
                    run_config.var_map_file_format = VarMapFileFormat.TEXT

        return run_config

    @staticmethod
    @contextmanager
    def ensure_sync_client(client: Optional[OpenAIClient]):
        if client:
            yield client
        else:
            with OpenAIClient(ClientMode.SYNC) as client:
                yield client

    @staticmethod
    @asynccontextmanager
    async def ensure_async_client(client: Optional[OpenAIClient]):
        if client:
            yield client
        else:
            async with OpenAIClient(ClientMode.ASYNC) as client:
                yield client

    @classmethod
    @abstractmethod
    def _run_with_client(
        cls,
        client: OpenAIClient,
        evaled_prompt: PromptType,
        stream: bool,
    ) -> PromptType: ...

    def run(
        self: PromptType,
        client: Optional[OpenAIClient] = None,
        run_config: RunConfig = DEFAULT_CONFIG,
        var_map: Optional[VarMapType] = None,
        **kwargs,
    ) -> PromptType:
        evaled_prompt, stream = self._prepare_run(run_config, var_map, kwargs)
        cls = type(self)
        with cls.ensure_sync_client(client) as client:
            return cls._run_with_client(client, evaled_prompt, stream)

    @classmethod
    @abstractmethod
    async def _arun_with_client(
        cls,
        client: OpenAIClient,
        evaled_prompt: PromptType,
        stream: bool,
    ) -> PromptType: ...

    async def arun(
        self: PromptType,
        client: Optional[OpenAIClient] = None,
        run_config: RunConfig = DEFAULT_CONFIG,
        var_map: Optional[VarMapType] = None,
        **kwargs,
    ) -> PromptType:
        evaled_prompt, stream = self._prepare_run(run_config, var_map, kwargs)
        cls = type(self)
        async with cls.ensure_async_client(client) as client:
            return await cls._arun_with_client(client, evaled_prompt, stream)

    @classmethod
    @abstractmethod
    def _stream_with_client(
        cls,
        requestor: Requestor,
        evaled_prompt: HandyPrompt,
    ) -> Generator[YieldType, None, None]:
        yield ...

    def stream(
        self,
        client: Optional[OpenAIClient] = None,
        run_config: RunConfig = DEFAULT_CONFIG,
        var_map: Optional[VarMapType] = None,
        **kwargs,
    ):
        evaled_prompt, _ = self._prepare_run(run_config, var_map, kwargs)
        cls = type(self)
        with cls.ensure_sync_client(client) as client:
            requestor = cls.get_requestor(client, evaled_prompt)
            yield from cls._stream_with_client(requestor, evaled_prompt)

    @classmethod
    @abstractmethod
    async def _astream_with_client(
        cls,
        requestor: Requestor,
        evaled_prompt: HandyPrompt,
    ) -> AsyncGenerator[YieldType, None]:
        yield ...

    @classmethod
    @abstractmethod
    def get_requestor(
        cls,
        client: OpenAIClient,
        evaled_prompt: HandyPrompt,
    ) -> Requestor: ...

    async def astream(
        self,
        client: Optional[OpenAIClient] = None,
        run_config: RunConfig = DEFAULT_CONFIG,
        var_map: Optional[VarMapType] = None,
        **kwargs,
    ) -> AsyncGenerator[YieldType, None]:
        evaled_prompt, _ = self._prepare_run(run_config, var_map, kwargs)
        cls = type(self)
        async with cls.ensure_async_client(client) as client:
            requestor = cls.get_requestor(client, evaled_prompt)
            async for item in cls._astream_with_client(requestor, evaled_prompt):
                yield item

    @classmethod
    @abstractmethod
    def _fetch_with_client(
        cls,
        requestor: Requestor,
        evaled_prompt: HandyPrompt,
    ) -> ResponseType: ...

    def fetch(
        self,
        client: Optional[OpenAIClient] = None,
        run_config: RunConfig = DEFAULT_CONFIG,
        var_map: Optional[VarMapType] = None,
        **kwargs,
    ):
        evaled_prompt, _ = self._prepare_run(run_config, var_map, kwargs)
        cls = type(self)
        with cls.ensure_sync_client(client) as client:
            requestor = cls.get_requestor(client, evaled_prompt)
            return cls._fetch_with_client(requestor, evaled_prompt)

    @classmethod
    @abstractmethod
    async def _afetch_with_client(
        cls,
        requestor: Requestor,
        evaled_prompt: HandyPrompt,
    ) -> ResponseType: ...

    async def afetch(
        self,
        client: Optional[OpenAIClient] = None,
        run_config: RunConfig = DEFAULT_CONFIG,
        var_map: Optional[VarMapType] = None,
        **kwargs,
    ):
        evaled_prompt, _ = self._prepare_run(run_config, var_map, kwargs)
        cls = type(self)
        async with cls.ensure_async_client(client) as client:
            requestor = cls.get_requestor(client, evaled_prompt)
            return await cls._afetch_with_client(requestor, evaled_prompt)

    @classmethod
    def _prepare_output_path(
        cls,
        output_path: PathType,
        start_time: datetime,
        template_filename: str,
        overwrite: bool,
    ):
        output_path = str(output_path).strip()
        p = Path(output_path)
        if p.is_dir() or output_path.endswith(("/")):
            # output_path wants to be a directory, append the default filename
            output_path = Path(output_path, template_filename)
        # format output_path with the current time
        output_path = start_time.strftime(str(output_path))
        output_path = Path(output_path)
        if not overwrite:
            org_path = output_path
            # append suffix until unique
            i = 1
            while output_path.exists():
                output_path = (
                    output_path.parent
                    / f"{org_path.stem}_{start_time.strftime(cls.TEMPLATE_TIMESTAMP)}_{i}{org_path.suffix}"
                )
                i += 1
        return output_path

    def _prepare_run(
        self,
        run_config: RunConfig,
        var_map: Optional[MutableMapping],
        kwargs: MutableMapping,
    ):
        # evaluate the prompt with the given run_config
        evaled_prompt = self.eval(run_config=run_config, var_map=var_map, **kwargs)
        evaled_run_config = evaled_prompt.run_config
        evaled_request = evaled_prompt.request

        # get the stream flag
        stream = evaled_request.get("stream", False)

        # verbose output
        if evaled_run_config.verbose:
            print("---", file=sys.stderr)
            print("NEW RUN")
            print(f"Start time: {datetime.now()}", file=sys.stderr)
            evaled_run_config.pretty_print()
            print("---", file=sys.stderr)

        # output the evaluated prompt to a file or a file descriptor
        # NOTE: should be done before loading the credential file
        if evaled_run_config.output_evaled_prompt_fd:
            evaled_prompt.dump(evaled_run_config.output_evaled_prompt_fd)
        elif evaled_run_config.output_evaled_prompt_path:
            evaled_prompt.dump_to(
                evaled_run_config.output_evaled_prompt_path, mkdir=True
            )

        # load the credential file
        if evaled_run_config.credential_path:
            if evaled_run_config.credential_type == CredentialType.ENV:
                load_dotenv(evaled_run_config.credential_path, override=True)
            elif evaled_run_config.credential_type in (
                CredentialType.JSON,
                CredentialType.YAML,
            ):
                with open(
                    evaled_run_config.credential_path, "r", encoding="utf-8"
                ) as fin:
                    if evaled_run_config.credential_type == CredentialType.JSON:
                        credential_dict = json_load(fin)
                    else:
                        credential_dict = yaml_load(fin)
                # do not overwrite the existing request arguments
                for key, value in credential_dict.items():
                    if key not in evaled_request:
                        evaled_request[key] = value
            else:
                raise ValueError(
                    f"unsupported credential type: {evaled_run_config.credential_type}"
                )

        return evaled_prompt, stream

    def _merge_non_data(
        self: PromptType, other: PromptType, inplace=False
    ) -> Tuple[MutableMapping, RunConfig]:
        if inplace:
            merge_dict(self.request, other.request, strategy=Strategy.ADDITIVE)
            self.run_config.merge(other.run_config, inplace=True)
            return self.request, self.run_config
        else:
            merged_request = merge_dict(
                {}, self.request, other.request, strategy=Strategy.ADDITIVE
            )
            merged_run_config = self.run_config.merge(other.run_config)
            return merged_request, merged_run_config

    @staticmethod
    def _filter_request(
        request: MutableMapping,
        run_config: RunConfig,
    ) -> MutableMapping:
        if run_config.record_request == RecordRequestMode.WHITELIST:
            if run_config.record_whitelist:
                request = {
                    key: value
                    for key, value in request.items()
                    if key in run_config.record_whitelist
                }
            else:
                request = {}
        elif run_config.record_request == RecordRequestMode.NONE:
            request = {}
        elif run_config.record_request == RecordRequestMode.ALL:
            pass
        else:
            # default: blacklist
            request = copy.deepcopy(request)
            real_blacklist = run_config.record_blacklist or DEFAULT_BLACKLIST
            for key in real_blacklist:
                request.pop(key, None)
        return request

    def _parse_var_map(self, run_config: RunConfig):
        var_map = {}
        if run_config.var_map_path:
            assert run_config.var_map_file_format is not None
            var_map = merge_dict(
                var_map,
                load_var_map(run_config.var_map_path, run_config.var_map_file_format),
                strategy=Strategy.REPLACE,
            )
        if run_config.var_map:
            var_map = merge_dict(var_map, run_config.var_map, strategy=Strategy.REPLACE)
        return var_map

    @abstractmethod
    def __add__(self: PromptType, other) -> PromptType: ...

    @staticmethod
    @contextmanager
    def open_output_path_fd(run_config: RunConfig):
        run_config.output_path = cast(PathType, run_config.output_path)
        # create the parent directory if it does not exist
        Path(run_config.output_path).parent.mkdir(parents=True, exist_ok=True)
        if (
            run_config.output_path_buffering is None
            or run_config.output_path_buffering == -1
        ):
            # default buffering
            with open(run_config.output_path, "w", encoding="utf-8") as fout:
                yield fout
        elif run_config.output_path_buffering == 0:
            # no buffering
            with open(run_config.output_path, "wb", buffering=0) as f_binary:
                yield io.TextIOWrapper(f_binary, encoding="utf-8", write_through=True)
        elif (
            isinstance(run_config.output_path_buffering, int)
            and run_config.output_path_buffering >= 1
        ):
            # 1 for line buffering, >= 2 for buffer size
            with open(
                run_config.output_path,
                "w",
                encoding="utf-8",
                buffering=run_config.output_path_buffering,
            ) as fout:
                fout.reconfigure(write_through=True)
                yield fout
        else:
            raise ValueError(
                f"unsupported output_path_buffering value: {run_config.output_path_buffering}"
            )

    @classmethod
    @contextmanager
    def open_fd(cls, run_config: RunConfig):
        if run_config.output_fd:
            # stream response to a file descriptor, no base_path
            yield run_config.output_fd
        elif run_config.output_path:
            # stream response to a file
            with cls.open_output_path_fd(run_config) as fout:
                yield fout
        else:
            yield None

    @classmethod
    @contextmanager
    def open_and_dump_frontmatter(cls, run_config: RunConfig, request: MutableMapping):
        with cls.open_fd(run_config) as fout:
            if fout:
                base_path = (
                    Path(run_config.output_path).parent.resolve()
                    if run_config.output_path
                    else None
                )
                fout.write(cls._dumps_frontmatter(request, run_config, base_path))
            yield fout


class ChatPrompt(HandyPrompt[ChatResponse, ChatChunk, List[Message]]):
    def __init__(
        self,
        messages: List[Message],
        request: Optional[MutableMapping] = None,
        meta: Optional[Union[MutableMapping, RunConfig]] = None,
        base_path: Optional[PathType] = None,
        response: Optional[ChatResponse] = None,
        requestor: Optional[Requestor] = None,
    ):
        super().__init__(messages, request, meta, base_path, response, requestor)

    @property
    def messages(self):
        """
        The underlying message data.
        """
        return self.data

    @messages.setter
    def messages(self, value):
        self.data = value

    @property
    def result_str(self) -> str:
        """
        Get the content string from the last message. If the content is
        a list, return the text part if it exists. Otherwise, return
        an empty string.
        """
        if len(self.messages) == 0:
            return ""
        content = self.messages[-1]["content"]
        if not content:
            return ""
        elif isinstance(content, list):
            for item in content:
                if "text" in item:
                    return item["text"]
            return ""
        else:
            return content

    @property
    def result_reasoning(self) -> str:
        """
        Get the reasoning content string from the last message. If it
        does not exist, return an empty string.
        """
        if len(self.messages) == 0:
            return ""
        elif "reasoning_content" in self.messages[-1]:
            content = self.messages[-1]["reasoning_content"]
            if content:
                return content
        return ""

    @staticmethod
    def _serialize_data(data: List[Message]) -> str:
        return converter.msgs2raw(data)

    def _eval_data(self, var_map) -> List[Message]:
        replaced = converter.msgs_replace_variables(
            self.messages, var_map, inplace=False
        )
        # replace local image URLs
        for msg in replaced:
            content = msg.get("content")
            if isinstance(content, list):
                for item in content:
                    try:
                        if item.get("type") == "image_url":
                            item = cast(ImageContentPart, item)
                            url = item["image_url"]["url"]
                            if url and url.startswith("file://"):
                                # replace the image URL with the actual image
                                item["image_url"]["url"] = file_uri_to_base64_image(
                                    url, self.base_path
                                )
                        elif item.get("type") == "input_audio":
                            item = cast(AudioContentPart, item)
                            data = item["input_audio"]["data"]
                            if data and data.startswith("file://"):
                                # replace the audio data with the actual audio
                                item["input_audio"]["data"] = file_uri_to_base64(
                                    data, self.base_path
                                )
                    except (KeyError, TypeError):
                        pass

        return replaced

    @classmethod
    def get_requestor(
        cls,
        client: OpenAIClient,
        evaled_prompt: HandyPrompt,
    ):
        return client.chat(messages=evaled_prompt.data, **evaled_prompt.request)

    @classmethod
    def _run_with_client(
        cls,
        client: OpenAIClient,
        evaled_prompt: HandyPrompt,
        stream: bool,
    ) -> ChatPrompt:
        run_config = evaled_prompt.run_config
        base_path = (
            Path(run_config.output_path).parent.resolve()
            if run_config.output_path
            else None
        )
        response = None
        requestor = cls.get_requestor(client, evaled_prompt)
        if stream:
            role = ""
            content = ""
            reasoning_content = ""
            tool_calls: List[ToolCall] = []
            for chunk in stream_chat_all(
                cls._stream_with_client(requestor, evaled_prompt)
            ):
                if chunk["role"] != role:
                    role = chunk["role"]
                if chunk["tool_call"]:
                    tool_calls.append(chunk["tool_call"])
                elif chunk["reasoning_content"]:
                    reasoning_content += chunk["reasoning_content"]
                elif chunk["content"]:
                    content += chunk["content"]
            msg: ResponseMessage = {
                "role": role,
                "content": content,
            }
            if tool_calls:
                msg["tool_calls"] = tool_calls
            if reasoning_content:
                msg["reasoning_content"] = reasoning_content
        else:
            response = cls._fetch_with_client(requestor, evaled_prompt)
            msg = response["choices"][0]["message"]
        return ChatPrompt(
            [msg],
            evaled_prompt.request,
            run_config,
            base_path,
            response=response,
            requestor=requestor,
        )

    @classmethod
    async def _arun_with_client(
        cls,
        client: OpenAIClient,
        evaled_prompt: HandyPrompt,
        stream: bool,
    ) -> ChatPrompt:
        run_config = evaled_prompt.run_config
        base_path = (
            Path(run_config.output_path).parent.resolve()
            if run_config.output_path
            else None
        )
        response = None
        requestor = cls.get_requestor(client, evaled_prompt)
        if stream:
            role = ""
            content = ""
            reasoning_content = ""
            tool_calls: List[ToolCall] = []
            async for chunk in astream_chat_all(
                cls._astream_with_client(requestor, evaled_prompt)
            ):
                if chunk["role"] != role:
                    role = chunk["role"]
                if chunk["tool_call"]:
                    tool_calls.append(chunk["tool_call"])
                elif chunk["reasoning_content"]:
                    reasoning_content += chunk["reasoning_content"]
                elif chunk["content"]:
                    content += chunk["content"]
            msg: ResponseMessage = {
                "role": role,
                "content": content,
            }
            if tool_calls:
                msg["tool_calls"] = tool_calls
            if reasoning_content:
                msg["reasoning_content"] = reasoning_content
        else:
            response = await cls._afetch_with_client(requestor, evaled_prompt)
            msg = response["choices"][0]["message"]
        return ChatPrompt(
            [msg],
            evaled_prompt.request,
            run_config,
            base_path,
            response=response,
            requestor=requestor,
        )

    @classmethod
    def _stream_with_client(
        cls,
        requestor: Requestor,
        evaled_prompt: HandyPrompt,
    ):
        run_config = evaled_prompt.run_config
        response = requestor.stream()
        with cls.open_and_dump_frontmatter(run_config, evaled_prompt.request) as fout:
            producer = trans_stream_chat(
                converter.consume_stream2fd(fout) if fout else echo_consumer()
            )
            next(producer)  # "prime" the coroutine
            for chat_chunk in response:
                ret = producer.send(chat_chunk)
                if run_config.on_chunk and ret:
                    run_config.on_chunk = cast(SyncHandlerChat, run_config.on_chunk)
                    role, content, reasoning_content, tool_call = ret
                    run_config.on_chunk(
                        {
                            "role": role,
                            "content": content,
                            "reasoning_content": reasoning_content,
                            "tool_call": tool_call,
                        }
                    )
                yield chat_chunk
            ret = producer.send(None)  # signal the end of the stream
            if run_config.on_chunk and ret:
                run_config.on_chunk = cast(SyncHandlerChat, run_config.on_chunk)
                role, content, reasoning_content, tool_call = ret
                run_config.on_chunk(
                    {
                        "role": role,
                        "content": content,
                        "reasoning_content": reasoning_content,
                        "tool_call": tool_call,
                    }
                )
            producer.close()

    @classmethod
    async def _astream_with_client(
        cls,
        requestor: Requestor,
        evaled_prompt: HandyPrompt,
    ):
        run_config = evaled_prompt.run_config
        response = await requestor.astream()
        with cls.open_and_dump_frontmatter(run_config, evaled_prompt.request) as fout:
            producer = trans_stream_chat(
                converter.consume_stream2fd(fout) if fout else echo_consumer()
            )
            next(producer)  # "prime" the coroutine
            async for chat_chunk in response:
                ret = producer.send(chat_chunk)
                if run_config.on_chunk and ret:
                    role, content, reasoning_content, tool_call = ret
                    if inspect.iscoroutinefunction(run_config.on_chunk):
                        await run_config.on_chunk(
                            {
                                "role": role,
                                "content": content,
                                "reasoning_content": reasoning_content,
                                "tool_call": tool_call,
                            }
                        )
                    else:
                        run_config.on_chunk = cast(SyncHandlerChat, run_config.on_chunk)
                        run_config.on_chunk(
                            {
                                "role": role,
                                "content": content,
                                "reasoning_content": reasoning_content,
                                "tool_call": tool_call,
                            }
                        )
                yield chat_chunk
            ret = producer.send(None)  # signal the end of the stream
            if run_config.on_chunk and ret:
                role, content, reasoning_content, tool_call = ret
                if inspect.iscoroutinefunction(run_config.on_chunk):
                    await run_config.on_chunk(
                        {
                            "role": role,
                            "content": content,
                            "reasoning_content": reasoning_content,
                            "tool_call": tool_call,
                        }
                    )
                else:
                    run_config.on_chunk = cast(SyncHandlerChat, run_config.on_chunk)
                    run_config.on_chunk(
                        {
                            "role": role,
                            "content": content,
                            "reasoning_content": reasoning_content,
                            "tool_call": tool_call,
                        }
                    )
            producer.close()

    @classmethod
    def _fetch_with_client(
        cls,
        requestor: Requestor,
        evaled_prompt: HandyPrompt,
    ):
        run_config = evaled_prompt.run_config
        response = requestor.fetch()
        cls._dump_fd_if_set(
            run_config, evaled_prompt.request, (response["choices"][0]["message"],)
        )
        return response

    @classmethod
    async def _afetch_with_client(
        cls,
        requestor: Requestor,
        evaled_prompt: HandyPrompt,
    ):
        run_config = evaled_prompt.run_config
        response = await requestor.afetch()
        cls._dump_fd_if_set(
            run_config, evaled_prompt.request, (response["choices"][0]["message"],)
        )
        return response

    def __add__(
        self: ChatPrompt,
        other: Union[
            str,
            Message,
            Iterable[Message],
            ChatPrompt,
        ],
    ):
        # support concatenation with string, list, dict or another ChatPrompt
        new_prompt = copy.deepcopy(self)
        new_prompt += other
        return new_prompt

    def __iadd__(
        self: ChatPrompt,
        other: Union[
            str,
            Message,
            Iterable[Message],
            ChatPrompt,
        ],
    ):
        # support concatenation with string, list, dict or another ChatPrompt
        if isinstance(other, str):
            self.add_message(content=other)
        elif isinstance(other, dict):
            self.messages.append(cast(Message, other))
        elif isinstance(other, Iterable):
            self.messages.extend(other)
        elif isinstance(other, ChatPrompt):
            # merge two ChatPrompt objects
            self.messages.extend(other.messages)
            self._merge_non_data(other, inplace=True)
        else:
            raise TypeError(
                f"unsupported operand type(s) for +: 'ChatPrompt' and '{type(other)}'"
            )
        return self

    def add_message(
        self,
        role: Union[
            Literal["system", "user", "assistant", "tool", "developer"], str
        ] = "user",
        content: Optional[
            Union[str, List[Union[TextContentPart, ImageContentPart, AudioContentPart]]]
        ] = None,
        tool_calls: Optional[List[ToolCall]] = None,
        tool_call_id: Optional[str] = None,
    ):
        msg = {"role": role, "content": content}
        if tool_calls is not None:
            msg["tool_calls"] = tool_calls
        if tool_call_id is not None:
            msg["tool_call_id"] = tool_call_id
        self.messages.append(cast(Message, msg))

    def add_content_part_to_message(
        self,
        content_part: Union[
            str,
            TextContentPart,
            ImageContentPart,
            AudioContentPart,
            FileContentPart,
            RefusalContentPart,
        ],
        message_index: int = -1,
    ):
        target_msg = cast(InputMessage, self.messages[message_index])
        assert target_msg["role"] != "tool", "cannot add content part to a tool message"
        if isinstance(target_msg["content"], str):
            old_content_part = content_part_text(target_msg["content"])
            target_msg["content"] = []
            target_msg["content"].append(old_content_part)
        if isinstance(content_part, str):
            content_part = content_part_text(content_part)
        target_msg["content"].append(content_part)  # type: ignore

    def add_text_to_message(
        self,
        text: str,
        message_index: int = -1,
    ):
        self.add_content_part_to_message(content_part=text, message_index=message_index)

    def add_image_url_to_message(
        self,
        url_or_base64: str,
        detail: Optional[str] = None,
        message_index: int = -1,
    ):
        self.add_content_part_to_message(
            content_part=content_part_image(url_or_base64, detail),
            message_index=message_index,
        )

    def add_input_audio_to_message(
        self,
        url_or_base64: str,
        format: str,
        message_index: int = -1,
    ):
        self.add_content_part_to_message(
            content_part=content_part_audio(url_or_base64, format),
            message_index=message_index,
        )

    def add_file_to_message(
        self,
        file_data: Optional[str] = None,
        file_id: Optional[str] = None,
        filename: Optional[str] = None,
        message_index: int = -1,
    ):
        self.add_content_part_to_message(
            content_part=content_part_file(
                file_data=file_data,
                file_id=file_id,
                filename=filename,
            ),
            message_index=message_index,
        )


class CompletionsPrompt(HandyPrompt[CompletionsResponse, CompletionsChunk, str]):
    def __init__(
        self,
        prompt: str,
        request: Optional[MutableMapping] = None,
        meta: Optional[Union[MutableMapping, RunConfig]] = None,
        base_path: Optional[PathType] = None,
        response: Optional[CompletionsResponse] = None,
        requestor: Optional[Requestor] = None,
    ):
        super().__init__(prompt, request, meta, base_path, response, requestor)

    @property
    def prompt(self) -> str:
        return self.data

    @prompt.setter
    def prompt(self, value: str):
        self.data = value

    def _eval_data(self, var_map) -> str:
        new_prompt = self.prompt
        for key, value in var_map.items():
            new_prompt = new_prompt.replace(key, value)
        return new_prompt

    @classmethod
    def get_requestor(
        cls,
        client: OpenAIClient,
        evaled_prompt: HandyPrompt,
    ):
        return client.completions(prompt=evaled_prompt.data, **evaled_prompt.request)

    @classmethod
    def _run_with_client(
        cls,
        client: OpenAIClient,
        evaled_prompt: HandyPrompt,
        stream: bool,
    ) -> CompletionsPrompt:
        run_config = evaled_prompt.run_config
        base_path = (
            Path(run_config.output_path).parent.resolve()
            if run_config.output_path
            else None
        )
        response = None
        requestor = cls.get_requestor(client, evaled_prompt)
        if stream:
            content = ""
            for text in stream_completions(
                cls._stream_with_client(requestor, evaled_prompt)
            ):
                content += text
        else:
            response = cls._fetch_with_client(requestor, evaled_prompt)
            content = response["choices"][0]["text"]
        return CompletionsPrompt(
            content,
            evaled_prompt.request,
            run_config,
            base_path,
            response=response,
            requestor=requestor,
        )

    @classmethod
    async def _arun_with_client(
        cls,
        client: OpenAIClient,
        evaled_prompt: HandyPrompt,
        stream: bool,
    ) -> CompletionsPrompt:
        run_config = evaled_prompt.run_config
        base_path = (
            Path(run_config.output_path).parent.resolve()
            if run_config.output_path
            else None
        )
        response = None
        requestor = cls.get_requestor(client, evaled_prompt)
        if stream:
            content = ""
            async for text in astream_completions(
                cls._astream_with_client(requestor, evaled_prompt)
            ):
                content += text
        else:
            response = await cls._afetch_with_client(requestor, evaled_prompt)
            content = response["choices"][0]["text"]
        return CompletionsPrompt(
            content,
            evaled_prompt.request,
            run_config,
            base_path,
            response=response,
            requestor=requestor,
        )

    @classmethod
    def _stream_with_client(
        cls,
        requestor: Requestor,
        evaled_prompt: HandyPrompt,
    ):
        run_config = evaled_prompt.run_config
        response = requestor.stream()
        with cls.open_and_dump_frontmatter(run_config, evaled_prompt.request) as fout:
            for chunk in response:
                try:
                    text = cast(str, chunk["choices"][0]["text"])
                    if fout:
                        fout.write(text)
                    if run_config.on_chunk:
                        run_config.on_chunk = cast(
                            SyncHandlerCompletions, run_config.on_chunk
                        )
                        run_config.on_chunk(text)
                except (KeyError, IndexError):
                    pass
                yield chunk

    @classmethod
    async def _astream_with_client(
        cls,
        requestor: Requestor,
        evaled_prompt: HandyPrompt,
    ):
        run_config = evaled_prompt.run_config
        response = await requestor.astream()
        with cls.open_and_dump_frontmatter(run_config, evaled_prompt.request) as fout:
            async for chunk in response:
                try:
                    text = cast(str, chunk["choices"][0]["text"])
                    if fout:
                        fout.write(text)
                    if run_config.on_chunk:
                        if inspect.iscoroutinefunction(run_config.on_chunk):
                            await run_config.on_chunk(text)
                        else:
                            run_config.on_chunk = cast(
                                SyncHandlerCompletions, run_config.on_chunk
                            )
                            run_config.on_chunk(text)
                except (KeyError, IndexError):
                    pass
                yield chunk

    @classmethod
    def _fetch_with_client(
        cls,
        requestor: Requestor,
        evaled_prompt: HandyPrompt,
    ):
        run_config = evaled_prompt.run_config
        response = requestor.fetch()
        cls._dump_fd_if_set(
            run_config, evaled_prompt.request, response["choices"][0]["text"]
        )
        return response

    @classmethod
    async def _afetch_with_client(
        cls,
        requestor: Requestor,
        evaled_prompt: HandyPrompt,
    ):
        run_config = evaled_prompt.run_config
        response = await requestor.afetch()
        cls._dump_fd_if_set(
            run_config, evaled_prompt.request, response["choices"][0]["text"]
        )
        return response

    def __add__(self: CompletionsPrompt, other: Union[str, CompletionsPrompt]):
        # support concatenation with string or another CompletionsPrompt
        new_prompt = copy.deepcopy(self)
        new_prompt += other
        return new_prompt

    def __iadd__(self: CompletionsPrompt, other: Union[str, CompletionsPrompt]):
        # support concatenation with string or another CompletionsPrompt
        if isinstance(other, str):
            self.add_text(other)
        elif isinstance(other, CompletionsPrompt):
            # merge two CompletionsPrompt objects
            self.add_text(other.prompt)
            self._merge_non_data(other, inplace=True)
        else:
            raise TypeError(
                f"unsupported operand type(s) for +: 'CompletionsPrompt' and '{type(other)}'"
            )
        return self

    def add_text(self, text: str):
        self.prompt += text


def loads(
    text: str,
    encoding: str = "utf-8",
    base_path: Optional[PathType] = None,
    cls: type[PromptType] = ChatPrompt,
) -> PromptType:
    """
    Load a HandyPrompt from a string, defaulting to a ChatPrompt.
    """
    return cls.loads(text, encoding, base_path)


def load(
    fd: IO[str],
    encoding: str = "utf-8",
    base_path: Optional[PathType] = None,
    cls: type[PromptType] = ChatPrompt,
) -> PromptType:
    """
    Load a HandyPrompt from an `IO[str]`
    (e.g., file descriptor) (must support `read()`).

    By default, assumes a ChatPrompt.
    """
    return cls.load(fd, encoding, base_path)


def load_from(
    path: PathType,
    encoding: str = "utf-8",
    cls: type[PromptType] = ChatPrompt,
) -> PromptType:
    """
    Load a HandyPrompt from a path, defaulting to a ChatPrompt.
    """
    return cls.load_from(path, encoding)


def dumps(prompt: HandyPrompt, base_path: Optional[PathType] = None) -> str:
    """
    Dump a HandyPrompt to a string.

    This is the same as calling `prompt.dumps()`.
    """
    return prompt.dumps(base_path)


def dump(
    prompt: HandyPrompt, fd: IO[str], base_path: Optional[PathType] = None
) -> None:
    """
    Dump a HandyPrompt to an `IO[str]` (e.g., file descriptor) (must support `write()`).

    This is the same as calling `prompt.dump()`.
    """
    return prompt.dump(fd, base_path)


def dump_to(
    prompt: HandyPrompt,
    path: PathType,
    mkdir: bool = False,
) -> None:
    """
    Dump a HandyPrompt to a path.

    This is the same as calling `prompt.dump_to()`.
    """
    return prompt.dump_to(path, mkdir=mkdir)


def load_var_map(
    path: PathType, format: VarMapFileFormat = VarMapFileFormat.TEXT
) -> Dict[str, str]:
    """
    Read all content that needs to be replaced in the prompt from a text file.
    """
    with open(path, "r", encoding="utf-8") as fin:
        if format in (VarMapFileFormat.JSON, VarMapFileFormat.YAML):
            return yaml_load(fin)
        content = fin.read()
    substitute_map = {}
    blocks = p_var_map.split(content)
    for idx in range(1, len(blocks), 2):
        key = blocks[idx]
        value = blocks[idx + 1]
        substitute_map[key] = value.strip()
    return substitute_map
