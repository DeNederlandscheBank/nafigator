"""Tests for `preprocessprocessor` module"""

import unittest
from nafigator.preprocessprocessor import convert_pdf, convert_docx
import pytest
import json
import camelot

unittest.TestLoader.sortTestMethodsUsing = None


# Needed documents: pdf with and without password

with open('tests/tests/convertor_output.json') as data_file:
    exp_output_dict = json.load(data_file)


@pytest.mark.parametrize("format, exp_output, params",
                         [('text', exp_output_dict['pdf2text'], {'fileDesc': {'author': 'anonymous'},
                                                                 'parse_tables_with_camelot': True}),
                          ('xml', exp_output_dict['pdf2xml'], {'fileDesc': {'author': 'anonymous'}}),
                          ('html', exp_output_dict['pdf2html'], {'fileDesc': {'author': 'anonymous'}})])
def test_convert_pdf(format: str,  exp_output: str, params: dict):
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
    assert params["pdfto" + format] == exp_output


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
