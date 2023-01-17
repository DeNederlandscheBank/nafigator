import unittest
import pandas as pd
import numpy as np

from nafigator.nafdocument import NafDocument


unittest.TestLoader.sortTestMethodsUsing = None

# Constants
NAF_VERSION = "testversion"
LANGUAGE = "testlanguage"
FILEDESC = {"title": "testtitle",
            "author": "testauthor",
            "creationtime": "testcreationtime",
            "filename": "testfilename",
            "filetype": "testfiletype",
            "pages": "testpages",
            }
PUBLIC = {"publicId": "testpublicId",
          "uri": "testuri",
          }

class TestNafDocument(unittest.TestCase):
    """
    The basic class that inherits unittest.TestCase
    """

    def test_generate(self):
        """
        This function tests whether the naf document initalization is done correctly
        input: etree._ElementTree + dict
        level: 2
        scenarios: check added features vs input
        """
        tmp = NafDocument()
        tmp.generate({"naf_version": NAF_VERSION,
                    "language": LANGUAGE,
                    "fileDesc": FILEDESC,
                    "public": PUBLIC,
                    }
                    )

        assert tmp.version == NAF_VERSION
        assert tmp.language == LANGUAGE
        assert tmp.header['fileDesc'] == FILEDESC
        assert tmp.header['public'] == PUBLIC


    def test_subelement(self):
        """
        This function tests whether subelement is added correctly
        input: etree._ElementTree OPTIONAL: [etree._Element, tag-string, data-dict, ignore-list]
        level: 0
        scenarios: check element input and ignore list
        #WARNING Does not override existing subelements
        """
        tmp = NafDocument().open(r"tests/tests/example.naf.xml")
        tmp.subelement(element=tmp.find("nafHeader"),tag="testtag",data={"testkey" : "testvalue"})
        tmp.subelement(element=tmp.find("nafHeader"),tag="testtag2",data={"testkey" : "testvalue",
                                                                        "testkey2" : "testvalue2"},
                                                                    attributes_to_ignore=['testkey'])
        assert tmp.find("nafHeader").find("testtag").tag == "testtag"
        assert tmp.find("nafHeader").find("testtag").attrib == {"testkey" : "testvalue"}
        assert tmp.find("nafHeader").find("testtag2").attrib == {"testkey2" : "testvalue2"}



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
        # TODO refactor nafigator code to support universal naf format
        """
        tmp = NafDocument().open(r"tests/tests/example.naf.xml")
        assert tmp.validate() == False



    def test_get_attributes(self):
        """
        test data of attributes output
        input: etree._ElementTree + dictlike OPTIONAL = [namespace-str, exclude-list]
        level: 0
        scenarios: check attributes vs input
        """
        pass

    def test_layer(self):
        """
        test layer output
        input: etree._ElementTree + str
        level: 0
        scenarios: check layer output
        """
        tmp = NafDocument().open(r"tests/tests/example.naf.xml")
        tmp.layer("testtag")
        tmp.layer("testtag2")
        elements = list(tmp.iter())
        assert elements[-2].tag == 'testtag'
        assert elements[-1].tag == 'testtag2'

    def test_add_filedesc_element(self):
        """
        test added filedescription element
        input: etree._ElementTree + dict
        level: 1
        scenarios: test elements vs input
        """
        pass

    def test_add_public_element(self):
        """
        test added public element
        input: etree._ElementTree + dict
        level: 1
        scenarios: test elements vs input
        """
        pass

    def test_add_wf_element(self):
        """
        test added wf element
        input: etree._ElementTree + wordform element + boolean
        level: 1
        scenarios: test elements vs input
        """
        pass


    def test_add_raw_text_element(self):
        """
        test added wf element
        input: etree._ElementTree + DependencyRelation + boolean
        level: 1
        scenarios: test elements vs input
        """
        pass

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
