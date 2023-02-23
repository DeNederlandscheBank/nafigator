# coding: utf-8

"""Preprocessor module."""

from pathlib import Path
import pdfminer
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter, XMLConverter, HTMLConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import BytesIO
from .const import ProcessorElement

import docx
import zipfile

try:
    from xml.etree.cElementTree import XML
except ImportError:
    from xml.etree.ElementTree import XML

import camelot as cm
import pdftopng


WORD_NAMESPACE = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
PARA = WORD_NAMESPACE + "p"
TEXT = WORD_NAMESPACE + "t"


def convert_docx(
    path: str = None,
    format: str = "text",
    codec: str = "utf-8",
    password: str = "",
    params: dict = None,
) -> str:
    """Function to convert docx to xml or text

    Args:
        path: location of the file to be converted
        format: text or xml
        codec: codec to be used to conversion
        password: password to be used for conversion
        params: the general params dict to store results

    Returns:
        str: the result of the conversion

    """

    if format == "text":
        stream = params.get("stream", None)
        if stream is not None:
            document = zipfile.ZipFile(stream)
        else:
            with open(path, "rb") as f:
                document = zipfile.ZipFile(f)
                text = document.read("word/document.xml")
        tree = XML(text)
        paragraphs = []
        for paragraph in tree.iter(PARA):
            texts = [node.text for node in paragraph.iter(TEXT) if node.text]
            if texts:
                paragraphs.append("".join(texts))
        text = "\n\n".join(paragraphs)

    elif format == "xml":
        stream = params.get("stream", None)
        if stream is not None:
            document = zipfile.ZipFile(stream)
        else:
            with open(path, "rb") as f:
                document = zipfile.ZipFile(f)
                text = document.read("word/document.xml")
                styles = document.read("word/styles.xml")  # not used yet

    params["docxto" + format] = text

    return None
