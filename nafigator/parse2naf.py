# coding: utf-8

"""Main module."""

from io import BytesIO
import os
import logging

from nafigator.nafparser import NafParser
from nafigator.preprocessor.pdf_converter import PDFConverter
from nafigator.preprocessor.docx_converter import DocxConverter
from nafigator.preprocessor.ocr_converter import OCRConverter
from nafigator.nafdocument.nafdocument import NafDocument


def generate_naf(
    path: str,
    engine: str,
    naf_version: str,
    stream: BytesIO = None,
    language: str = None,
    dtd_validation: bool = False,
    nlp=None,
    ocr: bool = False,
    params: dict = {}
) -> NafDocument:
    # Initialize parser
    parser = NafParser(
        engine=engine,
        language=language,
        naf_version=naf_version,
        dtd_validation=dtd_validation,
        nlp=nlp
    )

    # Define input as either string (doc path) or stream
    file = path
    if stream is not None:
        file = stream

    # Convert document to string or xml
    pdfdoc = None
    ext = os.path.splitext(path)[1]
    if ext == ".pdf":
        if ocr:
            text = OCRConverter().parse(file=file)
        else:
            pdfdoc = PDFConverter().parse(file=file)
            text = pdfdoc.text
    elif ext == ".docx":
        text = DocxConverter().parse(file=file)
    else:
        logging.error("document is not a pdf or docx and can not be parsed")
        return None

    # Generate NAF file
    return parser.generate_naf(
        input=path,
        stream=stream,
        text=text,
        pdfdoc=pdfdoc,
        params=params
    )
