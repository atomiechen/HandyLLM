__all__ = [
    'CacheManager'
]

from functools import wraps
from inspect import iscoroutinefunction
import json
from os import PathLike
from pathlib import Path
from typing import Callable, Collection, Iterable, Optional, TypeVar, Union, cast
from typing_extensions import ParamSpec
import yaml

from .types import PathType, StrHandler


def _load_output_files(
    files: Collection[Path], 
    convert_to: Optional[Union[Collection[Optional[StrHandler]], StrHandler]]
):
    all_files_exist = all(Path(file).exists() for file in files)
    if not all_files_exist:
        return None
    if convert_to is None:
        convert_to = (None,) * len(files)
    if not isinstance(convert_to, Collection):
        convert_to = (convert_to,) * len(files)
    if len(files) != len(convert_to):
        raise ValueError('The number of files and return types should be the same.')
    results = []
    for file, handle in zip(files, convert_to):
        with open(file, 'r', encoding='utf-8') as f:
            # determine the format according to the file suffix
            if file.suffix.endswith('.yaml') or file.suffix.endswith('.yml'):
                content = yaml.safe_load(f)
            elif file.suffix.endswith('.json'):
                content = json.load(f)
            else:
                content = f.read()
                if handle is not None:
                    content = handle(content)
        results.append(content)
    if len(results) == 1:
        return results[0]
    return tuple(results)

def _save_output_files(files: list[Path], results):
    if not isinstance(results, tuple):
        results = (results,)
    if len(files) != len(results):
        raise ValueError('The number of files and results should be the same.')
    for file, result in zip(files, results):
        file.parent.mkdir(parents=True, exist_ok=True)
        with open(file, 'w', encoding='utf-8') as f:
            # determine the format according to the file suffix
            if file.suffix.endswith('.yaml') or file.suffix.endswith('.yml'):
                yaml.dump(result, f, default_flow_style=False, allow_unicode=True)
            elif file.suffix.endswith('.json'):
                json.dump(result, f, ensure_ascii=False, indent=2)
            else:
                f.write(str(result))


P = ParamSpec("P")
R = TypeVar("R")

class CacheManager:
    def __init__(self, base_dir: PathType, enabled: bool = True, save_only: bool = False):
        self.base_dir = base_dir
        self.enabled = enabled
        self.save_only = save_only

    def cache(
        self, 
        func: Callable[P, R], 
        out: Union[PathType, Iterable[PathType]],
        enabled: Optional[bool] = None,
        save_only: Optional[bool] = None,
        convert_to: Optional[Union[Collection[Optional[StrHandler]], StrHandler]] = None,
    ) -> Callable[P, R]:
        '''
        Store the output of the function to the specified file. 
        The number of files must be the same as that of the results. 
        Supported file formats: yaml, json, txt
        '''
        if enabled is None:
            enabled = self.enabled
        if save_only is None:
            save_only = self.save_only
        if not enabled:
            return func
        if isinstance(out, str) or isinstance(out, PathLike):
            out = [out]
        full_files = [Path(self.base_dir, file) for file in out]
        if iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapped_func(*args: P.args, **kwargs: P.kwargs):
                if not save_only:
                    results = _load_output_files(full_files, convert_to)
                    if results is not None:
                        return cast(R, results)
                results = await func(*args, **kwargs)
                _save_output_files(full_files, results)
                return cast(R, results)
            return cast(Callable[P, R], async_wrapped_func)
        else:
            @wraps(func)
            def sync_wrapped_func(*args: P.args, **kwargs: P.kwargs):
                if not save_only:
                    results = _load_output_files(full_files, convert_to)
                    if results is not None:
                        return cast(R, results)
                results = func(*args, **kwargs)
                _save_output_files(full_files, results)
                return cast(R, results)
            return cast(Callable[P, R], sync_wrapped_func)

