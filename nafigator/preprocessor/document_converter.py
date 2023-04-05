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


class ConverterOutput(ABC):
    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def text(self, **kwargs):
        pass

    @abstractmethod
    def page_offsets(self, **kwargs):
        pass

    @abstractmethod
    def paragraph_offsets(self, **kwargs):
        pass

    @abstractmethod
    def write(self, **kwargs):
        pass
