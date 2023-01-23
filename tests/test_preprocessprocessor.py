"""Tests for `preprocessprocessor` module"""

import unittest
from nafigator.preprocessprocessor import convert_pdf, convert_docx
import pytest
import json
import camelot

unittest.TestLoader.sortTestMethodsUsing = None


# Needed documents: pdf with and without password

with open('tests/tests/pdf_convertor_output.json') as data_file:
    pdf_exp_output = json.load(data_file)


@pytest.mark.parametrize("format, params",
                         [('text', {'fileDesc': {'author': 'anonymous'},
                                    'parse_tables_with_camelot': True}),
                          ('xml', {'fileDesc': {'author': 'anonymous'}}),
                          ('html', {'fileDesc': {'author': 'anonymous'}})])
def test_convert_pdf(format: str, params: dict):
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
    path = 'tests/tests/example.pdf'
    convert_pdf(path=path, format=format, params=params)
    if params.get('parse_tables_with_camelot', False):
        assert type(params["pdftotables"]) == camelot.core.TableList
    output_key = "pdfto" + format
    assert params[output_key] == pdf_exp_output[output_key]


def test_convert_docx():
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
    pass
