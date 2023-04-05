# coding: utf-8

"""Preprocessor module."""

import regex
from dataclasses import dataclass
from io import BytesIO
from typing import Union
from lxml import etree
from pdfminer.converter import XMLConverter
from pdfminer.layout import LAParams
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.pdfpage import PDFPage
from nifigator.utils import replace_escape_characters
from .document_converter import DocumentConverter, ConverterOutput

# TODO streams component toevoegen
# TODO output lostrekken van de converter en standardiseren zodat docx hier ook gebruik van kan maken


@dataclass
class Offset:
    beginIndex: int
    endIndex: int


class PDFConverter(DocumentConverter):
    """Class to convert pdf to cleaned text"""

    def __init__(
        self,
        join_hyphenated_words: bool = True,
        ignore_control_characters: str = "[\x00-\x08\x0b-\x0c\x0e-\x1f]",
    ):
        self.join_hyphenated_words = join_hyphenated_words
        self.control_characters_to_ignore = regex.compile(ignore_control_characters)

    def parse(
        self,
        file: Union[str, BytesIO],
        codec: str = "utf-8",
        password: str = "",
        laparams: LAParams = LAParams(),
    ):
        """
        Function to convert pdf to xml or text
        Args:
            file: location or stream of the file to be converted
            codec: codec to be used to conversion
            password: password to be used for conversion
            laparams: laparams for the pdfminer.six parser
            join_hyphenated_words: Join 'hyhen-\\n ated wor- \\nds' to 'hyphenated words'
        Returns:
        """
        rsrcmgr = PDFResourceManager()
        retstr = BytesIO()
        device = XMLConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)

        if isinstance(file, str):
            file = open(file, "rb")

        interpreter = PDFPageInterpreter(rsrcmgr, device)
        maxpages = 0
        caching = True
        pagenos = set()
        for page in PDFPage.get_pages(
            file,
            pagenos,
            maxpages=maxpages,
            password=password,
            caching=caching,
            check_extractable=False,
        ):
            interpreter.process_page(page)

        # in case the file is opened, it is closed (a stream is not closed)
        if not isinstance(file, BytesIO):
            file.close()
        device.close()

        result = retstr.getvalue()
        retstr.close()

        parser = etree.XMLParser(ns_clean=True, recover=True, encoding="utf-8")
        tree = etree.fromstring(result, parser=parser)

        return PDFConverterOutput(tree)

    def open(self, input: Union[str, bytes]):
        """
        Function to open a PDFDocument in xml
        Args:
            input: the location of the PDFDocument in xml to be opened or a bytes object containing the file content
        """
        if isinstance(input, str):
            with open(input, "r", encoding="utf-8") as f:
                tree = etree.parse(f).getroot()
        elif type(input) == bytes:
            stream_data = BytesIO(input)
            tree = etree.parse(stream_data).getroot()
        else:
            raise TypeError(
                "invalid input, instead of bytes or string it is" + str(type(input))
            )
        return PDFConverterOutput(tree)

    def getstream(self) -> bytes:
        """
        Function to stream the PDFDocument in xml
        """
        output = BytesIO()
        super().write(output, encoding="utf-8", pretty_print=True, xml_declaration=True)
        return output


class PDFConverterOutput(ConverterOutput):
    """Class representing the XML output of the PDF parser"""

    def __init__(
        self,
        tree,
        join_hyphenated_words: bool = True,
        ignore_control_characters: str = "[\x00-\x08\x0b-\x0c\x0e-\x1f]"
    ):
        self.tree = tree
        self._format_tree(join_hyphenated_words, ignore_control_characters)

        # TODO: remove
        self.join_hyphenated_words = join_hyphenated_words
        self.ignore_control_characters = ignore_control_characters
        self.control_characters_to_ignore = regex.compile(ignore_control_characters)

    def _xml2text(self, tree: etree.Element, for_index: bool = False) -> str:
        """
        Function to extract text from xml
        Args:
            tree: the etree element
            for_index: determines if the result is used for the indices
        """
        text_list = []
        for text_element in tree.findall(".//text"):

            # Fix indices for ligatures
            text = text_element.text
            if for_index and (len(text) > 1):
                text = text[0]
            text_list.append(text)

        text_str = "".join(text_list)
        return text_str

    def _format_tree(self, join_hyphenated_words: bool, ignore_control_characters: str) -> None:
        """
        Function to format tree by removing control characters and (possible) joining hyphenated words
        Args:
            join_hyphenated_words: if hyphenated words should be joined
            ignore_control_characters: the control characters to ignore
        """
        # add and remove characters
        for textbox in self.tree.findall(".//page/textbox"):
            endline = etree.Element("text")
            endline.text = "\n"
            textbox.append(endline)

        for figure in self.tree.findall(".//figure"):
            for text_element in figure:
                if (text_element.text is None) or (text_element.text == "\n        "):
                    figure.remove(text_element)

        # convert to text
        text = self._xml2text(self.tree, True)

        # remove and fix control characters and hyphenated words from xml
        control_characters_to_ignore = regex.compile(ignore_control_characters)
        idx_control = [m.span() for m in control_characters_to_ignore.finditer(text)]

        idx_hyphens = []
        if join_hyphenated_words:
            _hyphens = "\u00AD\u058A\u05BE\u0F0C\u1400\u1806\u2010\u2011\u2012\u2e17\u30A0-"
            _hyphen_newline = regex.compile(
                r"(?<=\p{L})[" + _hyphens + "][ \t\u00a0\r]*\n{1,2}[ \t\u00a0]*(?=\\p{L})"
            )
            idx_hyphens = [m.span() for m in _hyphen_newline.finditer(text)]

        # get the indices of the characters to remove
        idx_all = idx_control + idx_hyphens
        idx2remove = []
        for r in idx_all:
            idx2remove += [*range(r[0], r[1])]

        # removal process
        for idx, text_element in enumerate(self.tree.findall(".//text")):
            if idx in idx2remove:
                textline = text_element.getparent()
                textline.remove(text_element)

    @property
    def text(self):
        """
        Property to extract text from PDFDocument
        """
        text = self._xml2text(self.tree)
        text = replace_escape_characters(text)
        return text

    @property
    def page_offsets(self):
        """
        Property to extract page offsets from PDFDocument
        """
        page_offsets = []
        text = ""
        for page in self.tree.findall(".//page"):
            page_start = len(text)
            text += self._xml2text(page)
            page_end = len(text)

            # append page offsets
            page_offsets.append(
                Offset(
                    beginIndex=page_start,
                    endIndex=page_end
                )
            )

        return page_offsets

    @property
    def paragraph_offsets(self):
        """
        Property to extract paragraph offsets from PDFDocument
        """
        paragraph_offsets = []
        text = ""
        for paragraph in self.tree.findall(".//page/textbox"):
            paragraph_start = len(text)
            text += self._xml2text(paragraph)
            paragraph_end = len(text)

            # append paragraph offsets
            paragraph_offsets.append(
                Offset(
                    beginIndex=paragraph_start,
                    endIndex=paragraph_end
                )
            )

        return paragraph_offsets

    def write(self, output: str) -> None:
        """Function to write an PDFDocument in xml
        Args:
            output: the location of the PDFDocument in xml to be stored
        """
        self.tree.getroottree().write(
            output, encoding="utf-8", pretty_print=True, xml_declaration=True
        )
