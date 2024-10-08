__all__ = ["CacheManager"]

from functools import wraps
from inspect import iscoroutinefunction
from os import PathLike
from pathlib import Path
from typing import Callable, Collection, Iterable, List, Optional, TypeVar, Union, cast
from typing_extensions import ParamSpec

from .types import PathType, StrHandler, StringifyHandler
from ._io import json_load, json_dump, yaml_load, yaml_dump


def _suffix_loader(file: Path):
    with open(file, "r", encoding="utf-8") as f:
        # determine the format according to the file suffix
        if file.suffix.endswith(".yaml") or file.suffix.endswith(".yml"):
            content = yaml_load(f)
        elif file.suffix.endswith(".json"):
            content = json_load(f)
        else:
            content = f.read()
        return content


def _suffix_dumper(file: Path, content):
    with open(file, "w", encoding="utf-8") as f:
        # determine the format according to the file suffix
        if file.suffix.endswith(".yaml") or file.suffix.endswith(".yml"):
            yaml_dump(content, f, default_flow_style=False)
        elif file.suffix.endswith(".json"):
            json_dump(content, f)
        else:
            f.write(str(content))


def _load_files(
    files: Collection[Path],
    load_method: Optional[Union[Collection[Optional[StrHandler]], StrHandler]],
    infer_from_suffix: bool,
):
    if load_method is None:
        load_method = (None,) * len(files)
    if not isinstance(load_method, Collection):
        load_method = (load_method,) * len(files)
    if len(files) != len(load_method):
        raise ValueError("The number of files and load_method should be the same.")
    results = []
    for file, handle in zip(files, load_method):
        if handle is not None:
            with open(file, "r", encoding="utf-8") as f:
                content = handle(f.read())
        elif infer_from_suffix:
            content = _suffix_loader(file)
        else:
            with open(file, "r", encoding="utf-8") as f:
                content = f.read()
        results.append(content)
    if len(results) == 1:
        return results[0]
    return tuple(results)


def _dump_files(
    results,
    files: List[Path],
    dump_method: Optional[
        Union[Collection[Optional[StringifyHandler]], StringifyHandler]
    ],
    infer_from_suffix: bool,
):
    if not isinstance(results, tuple):
        results = (results,)
    if len(files) != len(results):
        raise ValueError("The number of files and results should be the same.")
    if dump_method is None:
        dump_method = (None,) * len(files)
    if not isinstance(dump_method, Collection):
        dump_method = (dump_method,) * len(files)
    if len(files) != len(dump_method):
        raise ValueError("The number of files and dump_method should be the same.")
    for file, result, handler in zip(files, results, dump_method):
        file.parent.mkdir(parents=True, exist_ok=True)
        if handler is not None:
            with open(file, "w", encoding="utf-8") as f:
                f.write(handler(str(result)))
        elif infer_from_suffix:
            _suffix_dumper(file, result)
        else:
            with open(file, "w", encoding="utf-8") as f:
                f.write(str(result))


P = ParamSpec("P")
R = TypeVar("R")


class CacheManager:
    def __init__(
        self,
        base_dir: PathType,
        enabled: bool = True,
        only_dump: bool = False,
        only_load: bool = False,
    ):
        self.base_dir = base_dir
        self.enabled = enabled
        self.only_dump = only_dump
        self.only_load = only_load
        if only_dump and only_load:
            raise ValueError("only_dump and only_load cannot be True at the same time.")

    def cache(
        self,
        func: Callable[P, R],
        out: Union[PathType, Iterable[PathType]],
        enabled: Optional[bool] = None,
        only_dump: Optional[bool] = None,
        only_load: Optional[bool] = None,
        dump_method: Optional[
            Union[Collection[Optional[StringifyHandler]], StringifyHandler]
        ] = None,
        load_method: Optional[
            Union[Collection[Optional[StrHandler]], StrHandler]
        ] = None,
        infer_from_suffix: bool = True,
    ) -> Callable[P, R]:
        """
        Store the output of the function to the specified file.
        The number of files must be the same as that of the results.
        Supported file formats: yaml, json, txt
        """
        if enabled is None:
            enabled = self.enabled
        if only_dump is None:
            only_dump = self.only_dump
        if only_load is None:
            only_load = self.only_load
        if only_dump and only_load:
            raise ValueError("only_dump and only_load cannot be True at the same time.")
        if not enabled:
            return func
        if isinstance(out, str) or isinstance(out, PathLike):
            out = [out]
        full_files = [Path(self.base_dir, file) for file in out]
        if iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapped_func(*args: P.args, **kwargs: P.kwargs):
                if not only_dump:
                    non_exist_files = [
                        file for file in full_files if not Path(file).exists()
                    ]
                    if len(non_exist_files) == 0:
                        results = _load_files(
                            full_files, load_method, infer_from_suffix
                        )
                        return cast(R, results)
                    elif only_load:
                        raise FileNotFoundError(
                            f"Cache files not found: {non_exist_files}"
                        )
                results = await func(*args, **kwargs)
                _dump_files(results, full_files, dump_method, infer_from_suffix)
                return cast(R, results)

            return cast(Callable[P, R], async_wrapped_func)
        else:

            @wraps(func)
            def sync_wrapped_func(*args: P.args, **kwargs: P.kwargs):
                if not only_dump:
                    non_exist_files = [
                        file for file in full_files if not Path(file).exists()
                    ]
                    if len(non_exist_files) == 0:
                        results = _load_files(
                            full_files, load_method, infer_from_suffix
                        )
                        return cast(R, results)
                    elif only_load:
                        raise FileNotFoundError(
                            f"Cache files not found: {non_exist_files}"
                        )
                results = func(*args, **kwargs)
                _dump_files(results, full_files, dump_method, infer_from_suffix)
                return cast(R, results)

            return cast(Callable[P, R], sync_wrapped_func)

    def ensure_dumped(
        self,
        func: Callable[P, R],
        out: Union[PathType, Iterable[PathType]],
        dump_method: Optional[
            Union[Collection[Optional[StringifyHandler]], StringifyHandler]
        ] = None,
        infer_from_suffix: bool = True,
    ) -> Callable[P, None]:
        """
        Ensure the output of the original function is cached. Will not
        call the original function and will not load the files if they
        exist. The decorated function returns None.

        Example scenario:

        The decorated function can be called multiple times, but we only
        want to check files existence without loading them multiple times
        (which is the case of `cache` method).
        """
        if isinstance(out, str) or isinstance(out, PathLike):
            out = [out]
        full_files = [Path(self.base_dir, file) for file in out]
        if iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapped_func(*args: P.args, **kwargs: P.kwargs):
                non_exist_files = [
                    file for file in full_files if not Path(file).exists()
                ]
                if len(non_exist_files) > 0:
                    results = await func(*args, **kwargs)
                    _dump_files(results, full_files, dump_method, infer_from_suffix)

            return cast(Callable[P, None], async_wrapped_func)
        else:

            @wraps(func)
            def sync_wrapped_func(*args: P.args, **kwargs: P.kwargs):
                non_exist_files = [
                    file for file in full_files if not Path(file).exists()
                ]
                if len(non_exist_files) > 0:
                    results = func(*args, **kwargs)
                    _dump_files(results, full_files, dump_method, infer_from_suffix)

            return cast(Callable[P, None], sync_wrapped_func)
