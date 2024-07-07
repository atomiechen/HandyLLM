from __future__ import annotations
import sys
from enum import auto
from pathlib import Path
from typing import IO, Mapping, Optional
from dataclasses import dataclass, asdict, fields, replace

from mergedeep import merge as merge_dict, Strategy

from ._str_enum import AutoStrEnum
from .types import PathType, VarMapType, OnChunkType


class RecordRequestMode(AutoStrEnum):
    BLACKLIST = auto()  # record all request arguments except specified ones
    WHITELIST = auto()  # record only specified request arguments
    NONE = auto()  # record no request arguments
    ALL = auto()  # record all request arguments


class CredentialType(AutoStrEnum):
    # load environment variables from the credential file
    ENV = auto()
    # load the content of the file as request arguments
    JSON = auto()
    YAML = auto()


class VarMapFileFormat(AutoStrEnum):
    JSON = auto()
    YAML = auto()
    TEXT = auto()


@dataclass
class RunConfig:
    # record request arguments
    record_request: Optional[RecordRequestMode] = None  # default: RecordRequestMode.BLACKLIST
    record_blacklist: Optional[list[str]] = None  # default: DEFAULT_BLACKLIST
    record_whitelist: Optional[list[str]] = None
    # variable map
    var_map: Optional[VarMapType] = None
    # variable map file path
    var_map_path: Optional[PathType] = None
    # variable map file type: json, yaml, text
    var_map_file_format: Optional[VarMapFileFormat] = None  # default: guess from the file extension
    # callback for each chunk generated in stream mode of the response
    on_chunk: Optional[OnChunkType] = None
    # output the response to a file
    output_path: Optional[PathType] = None
    # buffering for opening the output file in stream mode: -1 for system default, 
    # 0 for unbuffered, 1 for line buffered, any other positive value for buffer size
    output_path_buffering: Optional[int] = None
    # output the response to a file descriptor
    output_fd: Optional[IO[str]] = None
    # output the evaluated prompt to a file or a file descriptor
    output_evaled_prompt_path: Optional[PathType] = None
    output_evaled_prompt_fd: Optional[IO[str]] = None
    # credential file path
    credential_path: Optional[PathType] = None
    # credential type: env, json, yaml
    # if env, load environment variables from the credential file
    # if json or yaml, load the content of the file as request arguments
    credential_type: Optional[CredentialType] = None  # default: guess from the file extension
    
    # verbose output to stderr
    verbose: Optional[bool] = None  # default: False
    
    def __setattr__(self, name: str, value: object):
        if name == "record_request":
            # validate record_request value
            if isinstance(value, str):
                if value not in RecordRequestMode:
                    raise ValueError(f"unsupported record_request value: {value}")
            elif isinstance(value, RecordRequestMode):
                value = value.value
            elif value is None:  # this field is optional
                pass
            else:
                raise ValueError(f"unsupported record_request value: {value}")
        elif name == "credential_type":
            # validate credential_type value
            if isinstance(value, str):
                if value not in CredentialType:
                    raise ValueError(f"unsupported credential_type value: {value}")
            elif isinstance(value, CredentialType):
                value = value.value
            elif value is None:  # this field is optional
                pass
            else:
                raise ValueError(f"unsupported credential_type value: {value}")
        elif name == "var_map_file_format":
            # validate var_map_file_format value
            if isinstance(value, str):
                if value not in VarMapFileFormat:
                    raise ValueError(f"unsupported var_map_file_format value: {value}")
            elif isinstance(value, VarMapFileFormat):
                value = value.value
            elif value is None:  # this field is optional
                pass
            else:
                raise ValueError(f"unsupported var_map_file_format value: {value}")
        super().__setattr__(name, value)
    
    def __len__(self):
        return len([f for f in fields(self) if getattr(self, f.name) is not None])
    
    @classmethod
    def from_dict(cls, obj: Mapping, base_path: Optional[PathType] = None):
        input_kwargs = {}
        for field in fields(cls):
            if field.name in obj:
                input_kwargs[field.name] = obj[field.name]
        # add base_path to path fields and convert to resolved path
        if base_path:
            for path_field in ("output_path", "output_evaled_prompt_path", "var_map_path", "credential_path"):
                if path_field in input_kwargs:
                    org_path = input_kwargs[path_field]
                    new_path = str(Path(base_path, org_path).resolve())
                    # retain trailing slash
                    if org_path.endswith(('/')):
                        new_path += '/'
                    input_kwargs[path_field] = new_path
        return cls(**input_kwargs)
    
    def pretty_print(self, file=sys.stderr):
        print("RunConfig:", file=file)
        for field in fields(self):
            value = getattr(self, field.name)
            if value is not None:
                print(f"  {field.name}: {value}", file=file)
    
    def to_dict(self, retain_object=False, base_path: Optional[PathType] = None) -> dict:
        # record and remove file descriptors
        tmp_output_fd = self.output_fd
        tmp_output_evaled_prompt_fd = self.output_evaled_prompt_fd
        tmp_on_chunk = self.on_chunk
        self.output_fd = None
        self.output_evaled_prompt_fd = None
        self.on_chunk = None
        # convert to dict
        obj = asdict(self, dict_factory=lambda x: { k: v for k, v in x if v is not None })
        # restore file descriptors
        self.output_fd = tmp_output_fd
        self.output_evaled_prompt_fd = tmp_output_evaled_prompt_fd
        self.on_chunk = tmp_on_chunk
        if retain_object:
            # keep file descriptors
            obj["output_fd"] = self.output_fd
            obj["output_evaled_prompt_fd"] = self.output_evaled_prompt_fd
            obj["on_chunk"] = self.on_chunk
        # convert path to relative path
        if base_path:
            for path_field in ("output_path", "output_evaled_prompt_path", "var_map_path", "credential_path"):
                if path_field in obj:
                    org_path = obj[path_field]
                    try:
                        new_path = str(Path(org_path).relative_to(base_path))
                        obj[path_field] = new_path
                    except ValueError:
                        # org_path is not under base_path, keep the original path
                        pass
        return obj

    def merge(self, other: RunConfig, inplace=False) -> RunConfig:
        # merge the RunConfig object with another RunConfig object
        # return a new RunConfig object if inplace is False
        if not inplace:
            new_run_config = replace(self)
        else:
            new_run_config = self
        for field in fields(RunConfig):
            v = getattr(other, field.name)
            if v is not None:
                if field.name == 'var_map':
                    if new_run_config.var_map is None:
                        new_run_config.var_map = {}
                    # merge the two var_map dicts in place
                    merge_dict(new_run_config.var_map, v, strategy=Strategy.REPLACE)
                else:
                    setattr(new_run_config, field.name, v)
        return new_run_config

