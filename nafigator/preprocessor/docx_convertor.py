# coding: utf-8

"""Preprocessor module."""

import zipfile
from io import BytesIO
from typing import Union
from .document_converter import DocumentConverter
import logging

try:
    from xml.etree.cElementTree import XML
except ImportError:
    from xml.etree.ElementTree import XML

WORD_NAMESPACE = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
PARA = WORD_NAMESPACE + "p"
TEXT = WORD_NAMESPACE + "t"


class DocxConverter(DocumentConverter):
    """Class to convert docx to cleaned text"""

    def __init__(self) -> None:
        pass

    def convert_docx(
        self,
        file: Union[str, BytesIO],
        format: str = "text"
    ) -> str:
        """Function to convert docx to xml or text

        Args:
            path: location of the file to be converted
            format: text or xml
            params: the general params dict to store results

        Returns:
            str: the result of the conversion
        """
        if format not in ["text", "xml"]:
            logging.error("The format should be 'text' or 'xml'")
            return ""

        if isinstance(file, str):
            with open(file, "rb") as f:
                document = zipfile.ZipFile(f)
        else:
            document = zipfile.ZipFile(file)

        text = document.read("word/document.xml")
        # styles = document.read("word/styles.xml")  # not used yet

        if format == "text":
            tree = XML(text)
            paragraphs = []
            for paragraph in tree.iter(PARA):
                texts = [node.text for node in paragraph.iter(TEXT) if node.text]
                if texts:
                    paragraphs.append("".join(texts))
            text = "\n\n".join(paragraphs)

        return text
