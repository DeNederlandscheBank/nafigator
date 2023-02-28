from abc import ABC, abstractmethod
from io import BytesIO
from typing import Union


class DocumentConverter(ABC):
    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def parse(self, file: Union[str, BytesIO], **kwargs):
        pass
