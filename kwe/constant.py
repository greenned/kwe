from enum import Enum


class StrEnum(Enum):
    def __str__(self):
        return self.value


class Color(StrEnum):
    header = "\033[94m"
    endc = "\033[0m"
    point = "\033[91m"
