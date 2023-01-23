"""Tests for `preprocessprocessor` module"""

import unittest
from nafigator.preprocessprocessor import convert_pdf, convert_docx
import json
import pytest

unittest.TestLoader.sortTestMethodsUsing = None

with open('tests/tests/docx_convertor_output.json') as data_file:
    docx_exp_output = json.load(data_file)


def test_convert_pdf():
    """
    This function converts a pdf file into text, html or xml.
    Input:
        path: location of the file to be converted
        format: html, text or xml
        codec: codec to be used to conversion
        password: password to be used for conversion
        params: the general params dict to store results
    Level: 0
    Scenarios:
        conversion to text
        conversion to html
        conversion to xml
    """
    pass


@pytest.mark.parametrize("format", ["text", "xml"])
def test_convert_docx(format: str):
    """
    This function converts a docx file into text or xml.
    Input:
        path: location of the file to be converted
        format: text or xml
        codec: codec to be used to conversion
        password: password to be used for conversion
        params: the general params dict to store results
    Level: Out of scope in refactoring phase 1
    """
    path = 'tests/tests/example.docx'
    params = {}
    convert_docx(path=path, format=format, params=params)
    key_output = "docxto" + format
    if isinstance(params[key_output], bytes):
        actual_output = params[key_output].decode("utf-8")
    else:
        actual_output = params[key_output]
    assert actual_output == docx_exp_output[key_output]
