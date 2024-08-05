from pathlib import Path
import yaml
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


def yaml_dump(*args, **kwargs):
    return yaml.dump(*args, Dumper=MySafeDumper, allow_unicode=True, **kwargs)


def yaml_load(stream):
    return yaml.safe_load(stream)

