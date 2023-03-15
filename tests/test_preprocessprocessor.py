# """Tests for `preprocessprocessor` module"""

# import unittest
# from nafigator.preprocessprocessor import convert_pdf, convert_docx
# import pytest
# import json
# import camelot

# unittest.TestLoader.sortTestMethodsUsing = None


# @pytest.fixture
# def docx_exp_output():
#     with open('tests/tests/docx_convertor_output.json') as data_file:
#         return json.load(data_file)


# @pytest.fixture
# def pdf_exp_output():
#     with open('tests/tests/pdf_convertor_output.json') as data_file:
#         return json.load(data_file)


# class TestConverter():
#     @pytest.mark.parametrize("format, params",
#                              [('text', {'fileDesc': {'author': 'anonymous'},
#                                         'parse_tables_with_camelot': True}),
#                               ('xml', {'fileDesc': {'author': 'anonymous'}}),
#                               ('html', {'fileDesc': {'author': 'anonymous'}})])
#     def test_convert_pdf(self, pdf_exp_output: dict, format: str, params: dict):
#         """
#         This function converts a pdf file into text, html or xml.
#         Input:
#             path: location of the file to be converted
#             format: html, text or xml
#             codec: codec to be used to conversion
#             password: password to be used for conversion
#             params: the general params dict to store results
#         Level: 0
#         Scenarios:
#             conversion to text
#             conversion to html
#             conversion to xml
#         """
#         path = 'tests/tests/example.pdf'
#         convert_pdf(path=path, format=format, params=params)
#         if params.get('parse_tables_with_camelot', False):
#             assert type(params["pdftotables"]) == camelot.core.TableList
#         output_key = "pdfto" + format
#         assert params[output_key] == pdf_exp_output[output_key]

#     @pytest.mark.parametrize("format", ["text", "xml"])
#     def test_convert_docx(self, docx_exp_output: dict, format: str):
#         """
#         This function converts a docx file into text or xml.
#         Input:
#             path: location of the file to be converted
#             format: text or xml
#             codec: codec to be used to conversion
#             password: password to be used for conversion
#             params: the general params dict to store results
#         Level: Out of scope in refactoring phase 1
#         """
#         path = 'tests/tests/example.docx'
#         params = {}
#         convert_docx(path=path, format=format, params=params)
#         key_output = "docxto" + format
#         actual_output = params[key_output]
#         if isinstance(params[key_output], bytes):
#             actual_output = params[key_output].decode("utf-8")

#         assert actual_output == docx_exp_output[key_output]
