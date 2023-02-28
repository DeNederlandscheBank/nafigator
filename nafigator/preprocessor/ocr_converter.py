# coding: utf-8

"""OCR module."""
import pdf2image
import pytesseract
from io import BytesIO
from typing import Union
from .document_converter import DocumentConverter


class OCRConverter(DocumentConverter):
    """Class to convert unreadable pdf to cleaned text"""

    def __init__(self) -> None:
        pass

    def parse(self, file: Union[str, BytesIO]) -> str:
        """Function to process ocr on pdf to generate text

        Source: https://stackoverflow.com/questions/29657237/tesseract-ocr-pdf-as-input

        Args:
            file: location of the file to be converted

        Returns:
            str: the result of the conversion

        """
        if isinstance(file, str):
            images = pdf2image.convert_from_path(file)
        else:
            images = pdf2image.convert_from_bytes(file.getvalue())

        text = [pytesseract.image_to_string(image) for _, image in enumerate(images)]

        return "\n".join(text)
