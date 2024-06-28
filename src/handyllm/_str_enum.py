from enum import Enum, EnumMeta


class StrEnumMeta(EnumMeta):
    def __contains__(cls, item):
        if isinstance(item, str):
            return item in iter(cls) # type: ignore
        return super().__contains__(item)


class AutoStrEnum(Enum, metaclass=StrEnumMeta):
    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        return name.lower()  # use lower case as the value
    
    def __eq__(self, other: object) -> bool:
        if isinstance(other, Enum):
            return self.value == other.value
        elif isinstance(other, str):
            return self.value == other.lower()
        return False

