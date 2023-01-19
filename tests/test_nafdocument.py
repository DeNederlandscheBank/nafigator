import unittest
from unittest.mock import patch, MagicMock, mock_open
import pytest
import datetime
from collections import namedtuple

from nafigator.nafdocument import NafDocument
from nafigator import nafdocument, EntityElement, TermElement, MultiwordElement

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
    wf_element = namedtuple("WfElemenwt","text id sent para page offset length xpath")
    return wf_element("testtext","testid","testsent","testpara","testpage","testoffset","testlength","testxpath")


@pytest.fixture
def doc():
    return NafDocument().open("tests/tests/example.naf.xml")

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
    def test_subelement(self, doc: NafDocument, data: dict, attributes_to_ignore: list):
        """
        This function tests whether subelement is added correctly
        input: etree._ElementTree OPTIONAL: [etree._Element, tag-string, data-dict, ignore-list]
        level: 0
        scenarios: check element input and ignore list
        #WARNING Does not override existing subelements
        """
        find_header = doc.find(nafdocument.NAF_HEADER)
        test_tag = "testtag"
        doc.subelement(
            element=find_header, 
            tag=test_tag, 
            data=data,
            attributes_to_ignore=attributes_to_ignore
        )

        data_without_ignore = {key: data[key] for key in data.keys() if key not in attributes_to_ignore}

        find_tag = find_header.find(test_tag)
        assert find_tag.tag == test_tag
        assert find_tag.attrib == data_without_ignore

    def test_add_processor_Element(self, doc: NafDocument):
        """
        This function tests whether processor element is added correctly
        input: etree._ElementTree + str + ProcessorElement
        level: 1
        scenarios: check element input and ignore list
        """
        test_layer = "test_layer"
        data = MagicMock()
        doc.add_processor_element(layer=test_layer, data=data)

        find_layer = doc.find(nafdocument.NAF_HEADER).find(f"./{nafdocument.LINGUISTIC_LAYER_TAG}[@layer='{test_layer}']")
        assert find_layer is not None
        assert find_layer.find(nafdocument.LINGUISTIC_OCCURRENCE_TAG).attrib == data

    def test_validate(self, doc: NafDocument):
        """
        test validate output
        input:etree._ElementTree
        level: 1 (uses utilsfunction load_dtd)
        scenarios: check xml string
        # TODO refactor nafigator code to support universal naf format. Also consider moving to integratin test
        """
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

        data = {
            "testkey": "testvalue", 
            "listkey": [], 
            "intkey": 1, 
            "floatkey": 1.0, 
            "excludekey": "excludevalue", 
            "datetimekey": datetime.datetime(2000, 1, 1, 9, 0),
            "nonekey": None,
        }       
        exp_output = {
            "testkey": "testvalue", 
            "intkey": str(1), 
            "floatkey": str(1.0), 
            "datetimekey": '2000-01-01T09:00:00UTC'
        }
        actual_output = doc.get_attributes(data, exclude = ["excludekey"])

        namedtuple_as_dict = namedtuple("namedtuple_as_dict", "testkey")
        data_not_dict = namedtuple_as_dict("testvalue")
        exp_output_not_dict =  {"testkey": "testvalue"}
        actual_output_not_dict = doc.get_attributes(data_not_dict)

        assert actual_output == exp_output
        assert actual_output_not_dict == exp_output_not_dict 

    def test_layer(self, doc: NafDocument):
        """
        test layer output
        input: etree._ElementTree + str
        level: 0
        scenarios: check layer output
        """
        test_tag = "testtag"
        test_tag2 = "testtag2"
        doc.layer(test_tag)
        doc.layer(test_tag2)
        elements = list(doc.iter())

        assert elements[-2].tag == test_tag
        assert elements[-1].tag == test_tag2

    def test_add_filedesc_element(self, doc: NafDocument):
        """
        test added filedescription element
        input: etree._ElementTree + dict
        level: 1
        scenarios: test elements vs input
        """
        data = {
            "title": "test",
            "author": "DreamWorks",
            "creationtime": "2000-01-01T09:00:00",
            "filename": "testfile",
            "filetype": "PDF",
            "page": "1"
        }

        doc.add_filedesc_element(data)
        
        assert doc.header.get("fileDesc") == data

    def test_add_public_element(self, doc: NafDocument, public_var: str):
        """
        test added public element
        input: etree._ElementTree + dict
        level: 1
        scenarios: test elements vs input
        """
        doc.add_public_element(public_var)
        assert doc.header['public'] == public_var        

    def test_add_wf_element(self, doc: NafDocument, wf_element_var):
        """
        test added wf element
        input: etree._ElementTree + wordform element + boolean
        level: 1
        scenarios: test elements vs input
        """
        doc = NafDocument().open(r"tests/tests/example.naf.xml")
        doc.add_wf_element(wf_element_var,True)

        attributes_to_ignore = "text"
        data = wf_element_var._asdict()
        data_without_ignore = {key: data[key] for key in data.keys() if key not in attributes_to_ignore}
        assert list(doc.layer("text").iter())[-1].attrib == data_without_ignore

    def test_add_raw_text_element(self, doc: NafDocument):
        """
        test added wf element
        input: etree._ElementTree + DependencyRelation + boolean
        level: 1
        scenarios: test elements vs input
        """        
        RawElement = namedtuple("RawElement","text")
        data = RawElement("This is raw text") 

        doc.add_raw_text_element(data)

        assert list(doc.iter())[-1].text == data.text

    @pytest.mark.parametrize('span,ext_refs', [
        ([], []),
        (["test_span"], ["test_ref"]),
    ])
    def test_add_entity_element(self, doc: NafDocument, span: list, ext_refs: list):
        """
        test added entity element
        input: etree._ElementTree + EntityElement + str + boolean
        level: 1
        scenarios: test elements vs input
        """
        test_id = "test_id"
        test_type = "test_type"
        data = EntityElement(
            id=test_id, 
            type=test_type,
            status=None,
            source=None,
            span=span,
            ext_refs=ext_refs,
            comment=None
        )

        naf_version = "test_version"
        comments = False
        doc.add_span_element = MagicMock()
        doc.add_external_reference_element = MagicMock()
        doc.add_entity_element(data, naf_version, comments)

        find_entities = doc.find(nafdocument.ENTITIES_LAYER_TAG).find(f"./{nafdocument.ENTITY_OCCURRENCE_TAG}[@id='test_id']")
        assert find_entities.attrib == {"id": test_id, "type": test_type}

        if span != []:
            doc.add_span_element.assert_called_once()
        if ext_refs != []:
            doc.add_external_reference_element.assert_called_once()

    @pytest.mark.parametrize('span,ext_refs', [
        ([], []),
        (["test_span"], ["test_ref"]),
        ])
    def test_add_term_element(self, doc: NafDocument, span: list, ext_refs: list):
        """
        test added term element
        input: etree._ElementTree + TermElement + str + boolean
        level: 2
        scenarios: test elements vs input
        """ 
        test_id = "test_id"
        test_type = "test_type"
        data = TermElement(
            id=test_id, 
            type=test_type,
            lemma = None,
            pos = None,
            morphofeat= None,
            netype = None,
            case = None,
            head = None,
            component_of=None,
            compound_type=None,
            span = span,
            ext_refs=ext_refs,
            comment=None
        )

        doc.add_span_element = MagicMock()
        doc.add_external_reference_element = MagicMock()
        doc.add_term_element(data, layer_to_attributes_to_ignore={},comments=False)

        find_terms = doc.find(nafdocument.TERMS_LAYER_TAG).find(f"./{nafdocument.TERM_OCCURRENCE_TAG}[@id='test_id']")
        assert find_terms.attrib == {"id": test_id, "type": test_type}

        if span != []:
            doc.add_span_element.assert_called_once()
        if ext_refs != []:
            doc.add_external_reference_element.assert_called_once()


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

    @pytest.mark.parametrize('span,components,lemma', [
        ([], [], []),
        (["test_span"],[{"test_id_component":"test_id_component", "test_type_component":"test_type_component"}], ["test_lemma"]),
        ])
    def test_add_multiword_element(self, doc: NafDocument, span: list, components: list, lemma: list):
        """
        test added multiword element
        input: etree._ElementTree + MultiwordElement
        level: 1
        scenarios: test elements vs input
        """
        test_id = "test_id"
        test_type = "test_type"

        data = MultiwordElement(
            id=test_id, 
            type=test_type,
            lemma=lemma,
            pos=None,
            morphofeat=None,
            case=None,
            status = None,
            components= components,
        )


        doc.add_span_element = MagicMock()
        doc.add_multiword_element(data)

        find_entities = doc.find(nafdocument.MULTIWORDS_LAYER_TAG).find(f"./{nafdocument.MULTIWORD_OCCURRENCE_TAG}[@id='test_id']")
        assert find_entities.attrib == {"id": test_id, "type": test_type}
        
        if span != []:
            doc.add_span_element.assert_called_once()

        


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
