import unittest
from unittest.mock import patch, MagicMock, mock_open
import pytest
import pandas as pd
import numpy as np
import datetime
from collections import namedtuple

from nafigator.nafdocument import NafDocument
from nafigator import nafdocument

unittest.TestLoader.sortTestMethodsUsing = None


@pytest.fixture
def version_var():
    return "testversion"


@pytest.fixture
def language_var():
    return "testlanguage"

@pytest.fixture
def filedesc_var():
    return {
        "title": "testtitle",
        "author": "testauthor",
        "creationtime": "testcreationtime",
        "filename": "testfilename",
        "filetype": "testfiletype",
        "pages": "testpages",
    }

@pytest.fixture
def public_var():
    return {
        "publicId": "testpublicId",
        "uri": "testuri",
    }

@pytest.fixture
def wf_element_var():
    return {
        "text" : "test_text",
        "id" : "test_id",
        "sent" : "test_sent",
        "para" : "test_para",
        "page" : "test_page",
        "offset" : "test_offset",
        "length" : "test_length",
        "xpath" : "test_xpath",
    }

class TestNafDocument():

    @pytest.mark.parametrize('language', ['language_var', None])
    def test_generate(self, version_var: str, filedesc_var: dict, public_var: dict, request: pytest.FixtureRequest, language: str):
        """
        This function tests whether the naf document initalization is done correctly
        input: etree._ElementTree + dict
        level: 2
        scenarios: check added features vs input
        """
        if language is not None:
            language = request.getfixturevalue(language)

        doc = NafDocument()
        doc.generate({
            "naf_version": version_var,
            "language": language,
            "fileDesc": filedesc_var,
            "public": public_var,
        })

        assert doc.version == version_var
        assert doc.language == language
        assert doc.header['fileDesc'] == filedesc_var
        assert doc.header['public'] == public_var

    @patch(f"{nafdocument.__name__}.etree")
    @pytest.mark.parametrize('input,expect_open', [
        ("tests/tests/example.naf.xml", True),
        (bytes("content", "utf-8"), False)
    ])
    def test_open(self, etree: MagicMock, input, expect_open: bool):
        """
        This function tests whether the naf document is opened correctly
        input: filepath (str) or bytes
        level: 2
        scenarios: check open and set root functions are called
        """
        parsed_tree = MagicMock()
        root = MagicMock()
        etree.parse.return_value = parsed_tree
        parsed_tree.getroot.return_value = root

        doc = NafDocument()
        doc._setroot = MagicMock()

        with patch("builtins.open", mock_open(read_data="data")) as mock_file:
            doc.open(input)

            # Check if the file was opened and root was set
            if expect_open:
                mock_file.assert_called_once_with(input, "r", encoding="utf-8")
            else:
                mock_file.assert_not_called()

            doc._setroot.assert_called_once_with(root)

    def test_open_error(self):
        """
        This function tests whether an exception is raised when the wrong type is trying to be opened
        input: int
        level: 2
        scenarios: check exception raised
        """
        with pytest.raises(TypeError):
            NafDocument().open(123)

    @pytest.mark.parametrize('data,attributes_to_ignore', [
        ({"testkey": "testvalue"}, []),
        ({"testkey2": "testvalue2", "testkeyignore": "testvalueignore"}, ["testkeyignore"])
    ])
    def test_subelement(self, data: dict, attributes_to_ignore: list):
        """
        This function tests whether subelement is added correctly
        input: etree._ElementTree OPTIONAL: [etree._Element, tag-string, data-dict, ignore-list]
        level: 0
        scenarios: check element input and ignore list
        #WARNING Does not override existing subelements
        """
        doc = NafDocument().open(r"tests/tests/example.naf.xml")
        naf_header = "nafHeader"
        test_tag = "testtag"
        doc.subelement(
            element=doc.find(naf_header), 
            tag=test_tag, 
            data=data,
            attributes_to_ignore=attributes_to_ignore
        )

        data_without_ignore = {key: data[key] for key in data.keys() if key not in attributes_to_ignore}

        assert doc.find(naf_header).find(test_tag).tag == test_tag
        assert doc.find(naf_header).find(test_tag).attrib == data_without_ignore


    def test_add_processor_Element(self):
        """
        This function tests whether processor element is added correctly
        input: etree._ElementTree + str + ProcessorElement
        level: 1
        scenarios: check element input and ignore list
        """
        pass

    def test_validate(self):
        """
        test validate output
        input:etree._ElementTree
        level: 1 (uses utilsfunction load_dtd)
        scenarios: check xml string
        # TODO refactor nafigator code to support universal naf format. Also consider moving to integratin test
        """
        doc = NafDocument().open(r"tests/tests/example.naf.xml")
        assert doc.validate() == False

    def test_get_attributes(self):
        """
        test data of attributes output
        input: etree._ElementTree + dictlike OPTIONAL = [namespace-str, exclude-list]
        level: 0
        scenarios: check attributes vs input
        # TODO namespace not included yet in test. Add once namespaces are 
        """
        doc = NafDocument()
        data = {"testkey": "testvalue", 
                "listkey": [], 
                "intkey": 1, 
                "floatkey": 1.0, 
                "excludekey": "excludevalue", 
                "datetimekey": datetime.datetime(2000, 1, 1, 9, 0)
                }       
        exp_output = {"testkey": "testvalue", 
                      "intkey": str(1), 
                      "floatkey": str(1.0), 
                      "datetimekey": '2000-01-01T09:00:00UTC'
                      }
        actual_output = doc.get_attributes(data, exclude = ["excludekey"])
        assert actual_output == exp_output

    def test_layer(self):
        """
        test layer output
        input: etree._ElementTree + str
        level: 0
        scenarios: check layer output
        """
        doc = NafDocument().open(r"tests/tests/example.naf.xml")
        doc.layer("testtag")
        doc.layer("testtag2")
        elements = list(doc.iter())
        assert elements[-2].tag == 'testtag'
        assert elements[-1].tag == 'testtag2'

    def test_add_filedesc_element(self):
        """
        test added filedescription element
        input: etree._ElementTree + dict
        level: 1
        scenarios: test elements vs input
        """

        doc = NafDocument().open(r"tests/tests/example.naf.xml")
        
        data = {"title": "test",
                "author": "DreamWorks",
                "creationtime": "2000-01-01T09:00:00",
                "filename": "testfile",
                "filetype": "PDF",
                "page": "1"
        }

        doc.add_filedesc_element(data)
        
        assert doc.header.get("fileDesc") == data

    def test_add_public_element(self, public_var):
        """
        test added public element
        input: etree._ElementTree + dict
        level: 1
        scenarios: test elements vs input
        """
        doc = NafDocument().open(r"tests/tests/example.naf.xml")
        doc.add_public_element(public_var)
        assert doc.header['public'] == public_var        

    def test_add_wf_element(self):
        """
        test added wf element
        input: etree._ElementTree + wordform element + boolean
        level: 1
        scenarios: test elements vs input
        """
        # doc = NafDocument().open(r"tests/tests/example.naf.xml")
        # wf = doc.subelement(
        #     element=doc.layer("text"),
        #     tag="wf",
        #     data=WF_ELEMENT,
        #     attributes_to_ignore=["text"],
        # ) 
        
        # # fails on dict and on element input
        # doc.add_wf_element(wf,True)
        pass

    def test_add_raw_text_element(self):
        """
        test added wf element
        input: etree._ElementTree + DependencyRelation + boolean
        level: 1
        scenarios: test elements vs input
        """

        doc = NafDocument().open(r"tests/tests/example.naf.xml")
        
        RawElement = namedtuple("RawElement","text")
        data = RawElement("This is raw text") 

        doc.add_raw_text_element(data)

        assert list(doc.iter())[-1].text == data.text

    def test_add_entity_element(self):
        """
        test added entity element
        input: etree._ElementTree + EntityElement + str + boolean
        level: 1
        scenarios: test elements vs input
        """
        pass

    def test_add_term_element(self):
        """
        test added term element
        input: etree._ElementTree + TermElement + str + boolean
        level: 2
        scenarios: test elements vs input
        """
        pass

    def test_add_chunk_element(self):
        """
        test added chunk element
        input: etree._ElementTree + ChunkElement + boolean
        level: 2
        scenarios: test elements vs input
        """
        pass

    def test_add_span_element(self):
        """
        test added span element
        input: etree._ElementTree + tree._ElementTree(2) + dictlike OPTIONAL [comments-boolean, naf_version str]
        level: 1
        scenarios: test elements vs input
        """
        pass

    def test_add_external_reference_element(self):
        """
        test added external reference element
        input: etree._ElementTree + tree._ElementTree(2) + list
        level: 1
        scenarios: test elements vs input
        """
        pass

    def test_add_multiword_element(self):
        """
        test added multiword element
        input: etree._ElementTree + MultiwordElement
        level: 1
        scenarios: test elements vs input
        """
        pass

    def test_add_formats_copy_element(self):
        """
        test added formats copy element
        input: etree._ElementTree + src str + formats str
        level: 0
        scenarios: test elements vs input
        """
        pass

    def test_add_formats_element(self):
        """
        test added formats element
        input: etree._ElementTree + src str + formats str + bool Optional: [camelot.core.TableList]
        level: 1
        scenarios: test elements vs input
        """
        pass
