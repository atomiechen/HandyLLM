import json
from pathlib import Path
import yaml
from functools import wraps

from .response import DictProxy


# add multi representer for Path, for YAML serialization
class MySafeDumper(yaml.SafeDumper):
    pass


MySafeDumper.add_multi_representer(
    Path, lambda dumper, data: dumper.represent_str(str(data))
)
MySafeDumper.add_multi_representer(
    DictProxy, lambda dumper, data: dumper.represent_dict(data)
)


@wraps(yaml.dump)
def yaml_dump(*args, **kwargs):
    kwargs.setdefault("Dumper", MySafeDumper)
    kwargs.setdefault("allow_unicode", True)
    return yaml.dump(*args, **kwargs)


@wraps(yaml.safe_load)
def yaml_load(*args, **kwargs):
    return yaml.safe_load(*args, **kwargs)


@wraps(json.dump)
def json_dump(*args, **kwargs):
    kwargs.setdefault("ensure_ascii", False)
    kwargs.setdefault("indent", 2)
    return json.dump(*args, **kwargs)


@wraps(json.dumps)
def json_dumps(*args, **kwargs):
    kwargs.setdefault("ensure_ascii", False)
    kwargs.setdefault("indent", 2)
    return json.dumps(*args, **kwargs)


@wraps(json.load)
def json_load(*args, **kwargs):
    return json.load(*args, **kwargs)


@wraps(json.loads)
def json_loads(*args, **kwargs):
    return json.loads(*args, **kwargs)
