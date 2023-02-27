# coding: utf-8

"""naf document."""

from lxml import etree
from typing import Union
import datetime
import logging
import io
from .const import (
    NAF_VERSION_TO_DTD,
    FILEDESC_ELEMENT_TAG,
    PUBLIC_ELEMENT_TAG,
    TERMS_LAYER_TAG,
    TERM_OCCURRENCE_TAG,
    DEPS_LAYER_TAG,
    DEP_OCCURRENCE_TAG,
    TEXT_LAYER_TAG,
    TEXT_OCCURRENCE_TAG,
    ENTITIES_LAYER_TAG,
    ENTITY_OCCURRENCE_TAG,
    CHUNKS_LAYER_TAG,
    CHUNK_OCCURRENCE_TAG,
    MULTIWORDS_LAYER_TAG,
    MULTIWORD_OCCURRENCE_TAG,
    COMPONENT_OCCURRENCE_TAG,
    RAW_LAYER_TAG,
    NAF_HEADER,
    SPAN_OCCURRENCE_TAG,
    TARGET_OCCURRENCE_TAG,
    EXT_REFS_OCCURRENCE_TAG,
    EXT_REF_OCCURRENCE_TAG,
    LINGUISTIC_LAYER_TAG,
    LINGUISTIC_OCCURRENCE_TAG,
    PREFIX_NAF_BASE
)
from .nafelements import (
    ProcessorElement,
    DependencyRelation,
    WordformElement,
    EntityElement,
    TermElement,
    MultiwordElement,
    ChunkElement,
    RawElement
)
from nafigator.utils import (
    load_dtd,
    prepare_comment_text
)

namespaces = {
    "dc": "http://purl.org/dc/elements/1.1/",
    # "naf-base": "https://dnb.nl/naf-Base/elements/1.0/",
}


def QName(prefix: str = None, name: str = None):
    """ """
    # currently no namespaces used
    # qname = etree.QName('{'+namespaces[prefix]+'}'+name, name)
    return name


def open_naf(input: Union[str, bytes]):
    """Function to open a NafDocument

    Args:
        input: the location of the NafDocument to be opened or a bytes object containing the file content

    Returns:
        NafDocument: the NAF document that is opened

    """
    if isinstance(input, str):
        with open(input, "r", encoding="utf-8") as f:
            root = etree.parse(f).getroot()
    elif type(input) == bytes:
        stream_data = io.BytesIO(input)
        root = etree.parse(stream_data).getroot()
    else:
        raise TypeError("invalid input, instead of bytes or string it is" + str(type(input)))

    return NafDocument(root=root)


class NafDocument(etree._ElementTree):
    """The NafDocument class (subclass of an etree.elementtree)"""

    def __init__(
        self,
        root: etree.Element = etree.Element("NAF", nsmap=namespaces),
        naf_version: str = None,
        language: str = None,
        filedesc_elem: dict = {},
        public_elem: dict = {}
    ) -> None:
        """Initialize a NafDocument"""
        self._setroot(root)
        self.add_nafHeader()
        self.set_version(naf_version)
        self.set_language(language)
        self.add_filedesc_element(filedesc_elem)
        self.add_public_element(public_elem)

    def write(self, output: str) -> None:
        """Function to write a NafDocument

        Args:
            output: the location of the NafDocument to be stored

        Returns:
            None

        """
        super().write(output, encoding="utf-8", pretty_print=True, xml_declaration=True)

    def getstream(self) -> bytes:
        """
        Function to stream the NafDocument
        Returns: Bytesstream of the NafDocument
        """
        output = io.BytesIO()
        super().write(output, encoding="utf-8", pretty_print=True, xml_declaration=True)
        return output

    @property
    def version(self):
        """Returns version of the NAF document"""
        return self.getroot().get("version")

    @property
    def language(self):
        """Returns language of the NAF document"""
        return self.getroot().get("{http://www.w3.org/XML/1998/namespace}lang")

    @property
    def header(self):
        """Returns header of the NAF document as a dict"""
        header = dict()
        ling_proc = list()
        for child in self.find(NAF_HEADER):
            if child.tag == "fileDesc":
                header["fileDesc"] = dict(child.attrib)
            if child.tag == "public":
                header["public"] = dict(child.attrib)
            if child.tag == LINGUISTIC_LAYER_TAG:
                header_data = dict(child.attrib)
                lp = list()
                for child2 in child:
                    if child2.tag == LINGUISTIC_OCCURRENCE_TAG:
                        lp.append(child2.attrib)
                header_data["lps"] = lp
                ling_proc.append(header_data)
        header[LINGUISTIC_LAYER_TAG] = ling_proc
        return header

    @property
    def layers(self):
        """Returns layers of the NAF document"""
        return [
            TEXT_LAYER_TAG,
            TERMS_LAYER_TAG,
            ENTITIES_LAYER_TAG,
            DEPS_LAYER_TAG,
            RAW_LAYER_TAG,
            MULTIWORDS_LAYER_TAG
        ]

    @property
    def raw(self):
        """Returns raw layer of the NAF document as string"""
        return self.find(RAW_LAYER_TAG).text

    @property
    def deps(self):
        """Returns dependencies layer of the NAF document as list"""
        return [
            dep.attrib
            for dep in self.findall(DEPS_LAYER_TAG + "/" + DEP_OCCURRENCE_TAG)
        ]

    @property
    def text(self):
        """Returns text layer of the NAF document as list of dicts"""
        return [
            dict({"text": wf.text}, **dict(wf.attrib))
            for wf in self.findall(TEXT_LAYER_TAG + "/" + TEXT_OCCURRENCE_TAG)
        ]

    @property
    def terms(self):
        """Returns terms layer of the NAF document as list of dicts"""
        terms = list()
        for child in self.findall(TERMS_LAYER_TAG + "/" + TERM_OCCURRENCE_TAG):
            term_data = dict(child.attrib)
            for child2 in child:
                if child2.tag == SPAN_OCCURRENCE_TAG:
                    span = [
                        child3.attrib
                        for child3 in child2
                        if child3.tag == TARGET_OCCURRENCE_TAG
                    ]
                    term_data["span"] = span
                if child2.tag == EXT_REFS_OCCURRENCE_TAG:
                    ext_refs = list()
                    for child3 in child2:
                        if child3.tag == EXT_REF_OCCURRENCE_TAG:
                            ext_refs.append(child3.attrib)
                    entity_data["ext_refs"] = ext_refs
            terms.append(term_data)
        return terms

    @property
    def multiwords(self):
        """Returns multiword layer of the NAF document as list of dicts"""
        mw = list()
        for child in self.findall(
            MULTIWORDS_LAYER_TAG + "/" + MULTIWORD_OCCURRENCE_TAG
        ):
            mw_data = dict(child.attrib)
            com = list()
            for child2 in child:
                if child2.tag == COMPONENT_OCCURRENCE_TAG:
                    com_data = dict(child2.attrib)
                    for child3 in child2:
                        if child3.tag == SPAN_OCCURRENCE_TAG:
                            span = [
                                child4.attrib
                                for child4 in child3
                                if child4.tag == TARGET_OCCURRENCE_TAG
                            ]
                            com_data["span"] = span
                        if child3.tag == EXT_REFS_OCCURRENCE_TAG:
                            ext_refs = list()
                            for child4 in child3:
                                if child4.tag == EXT_REF_OCCURRENCE_TAG:
                                    ext_refs.append(child4.attrib)
                            com_data["ext_refs"] = ext_refs
                    com.append(com_data)
            mw_data["components"] = com
            mw.append(mw_data)
        return mw

    @property
    def entities(self):
        """Returns entities layer of the NAF document as list of dicts"""
        entities = list()
        for child in self.findall(ENTITIES_LAYER_TAG + "/" + ENTITY_OCCURRENCE_TAG):
            entity_data = dict(child.attrib)
            spans = list()
            ext_refs = list()
            for child2 in child:
                if child2.tag == SPAN_OCCURRENCE_TAG:
                    span = list()
                    for child3 in child2:
                        if child3.tag == etree.Comment:
                            entity_data["text"] = child3.text
                        elif child3.tag == TARGET_OCCURRENCE_TAG:
                            span.append(child3.attrib)
                    spans.append(span)
                if child2.tag == EXT_REFS_OCCURRENCE_TAG:
                    ext_ref = list()
                    for child3 in child2:
                        if child3.tag == EXT_REF_OCCURRENCE_TAG:
                            ext_ref.append(child3.attrib)
                    ext_refs.append(ext_ref)
            if ext_refs != []:
                entity_data["ext_refs"] = ext_refs
            if spans != []:
                if len(spans) == 1:
                    entity_data["span"] = spans[0]
                else:
                    entity_data["spans"] = spans
            entities.append(entity_data)
        return entities

    @property
    def sentences(self):
        """Returns sentences of the NAF document as list of dicts"""
        word2term = {
            item["id"]: term["id"] for term in self.terms for item in term["span"]
        }
        last_sentence = list()
        sentences = list()
        sentence_list = list()
        sent_num = 1
        pages = set()
        para = set()
        span = list()
        terms = list()
        for item in self.text:
            if item["sent"] == str(sent_num):
                sentence_list.append(item["text"])
                span.append({"id": item["id"]})
                if item["id"] in word2term.keys():
                    terms.append({"id": word2term.get(item["id"])})
                pages.add(item.get("page", "0"))
                para.add(item.get("para", "0"))
                last_sentence = [item["sent"]]
            else:
                sentences.append(
                    {
                        "text": " ".join(sentence_list),
                        "para": list(para),
                        "page": list(pages),
                        "span": span,
                        "terms": terms,
                        "sent": last_sentence
                    }
                )
                sentence_list = list([item["text"]])
                pages = set()
                para = set()
                span = list()
                terms = list()
                span.append({"id": item["id"]})
                if item["id"] in word2term.keys():
                    terms.append({"id": word2term.get(item["id"])})
                pages.add(item.get("page", "0"))
                para.add(item.get("para", "0"))
                last_sentence = [item["sent"]]
                sent_num += 1
        if sent_num >= 1:
            if sentence_list != []:
                sentences.append(
                    {
                        "text": " ".join(sentence_list),
                        "para": list(para),
                        "page": list(pages),
                        "span": span,
                        "terms": terms,
                        "sent": last_sentence
                    }
                )
        return sentences

    @property
    def paragraphs(self):
        """Returns paragraphs of the NAF document as list of dicts"""

        # return empty list if no para attributes are included
        if any(["para" not in item.keys() for item in self.text]):
            return []

        word2term = {
            item["id"]: term["id"] for term in self.terms for item in term["span"]
        }
        paragraphs = list()
        paragraph_list = list()
        sentence_list = list()
        para_num = 1
        pages = set()
        para = set()
        span = list()
        terms = list()
        for item in self.text:
            if item["para"] == str(para_num):
                paragraph_list.append(item["text"])
                span.append({"id": item["id"]})
                if item["id"] in word2term.keys():
                    terms.append({"id": word2term.get(item["id"])})
                pages.add(item.get("page", "0"))
                para.add(item.get("para", "0"))
                if item.get("sent", "0") not in sentence_list:
                    sentence_list.append(item.get("sent", "0"))
            else:
                paragraphs.append(
                    {
                        "text": " ".join(paragraph_list),
                        "para": list(para),
                        "page": list(pages),
                        "span": span,
                        "terms": terms,
                        "sent": sentence_list
                    }
                )
                paragraph_list = list([item["text"]])
                sentence_list = list()
                pages = set()
                para = set()
                span = list()
                terms = list()
                span.append({"id": item["id"]})
                if item["id"] in word2term.keys():
                    terms.append({"id": word2term.get(item["id"])})
                pages.add(item.get("page", "0"))
                para.add(item.get("para", "0"))
                if item.get("sent", "0") not in sentence_list:
                    sentence_list.append(item.get("sent", "0"))
                para_num += 1
        if para_num > 1:
            paragraphs.append(
                {
                    "text": " ".join(paragraph_list),
                    "para": list(para),
                    "page": list(pages),
                    "span": span,
                    "terms": terms,
                    "sent": sentence_list
                }
            )
        return paragraphs

    def __getattr__(self, name):
        """Return custom made layer of the NAF document"""
        layer = self.find(name)
        if layer is not None:
            l = list()
            for item in layer:
                if item.text is not None:
                    item_data = dict({"text": item.text}, **dict(item.attrib))
                else:
                    item_data = dict(item.attrib)
                for span in item:
                    if span.tag == SPAN_OCCURRENCE_TAG:
                        targets = list()
                        for child3 in span:
                            if child3.tag == etree.Comment:
                                entity_data["text"] = child3.text
                            elif child3.tag == TARGET_OCCURRENCE_TAG:
                                targets.append(child3.attrib)
                    item_data["span"] = targets
                l.append(item_data)
            return l
        return super().name

    def set_language(self, language: str):
        """Set language of the NAF document"""
        if language is not None:
            self.getroot().set("{http://www.w3.org/XML/1998/namespace}lang", language)

    def set_version(self, version: str):
        """Set version of the NAF document"""
        if version is not None:
            self.getroot().set("version", version)

    def tree2string(self, byte: bool = False):
        """Return xml string of the NAF document"""
        xml_string = etree.tostring(
            self, pretty_print=True, xml_declaration=True, encoding="utf-8"
        )
        if byte:
            return xml_string
        else:
            return xml_string.decode("utf-8")

    def validate(self):
        """Validate xml string of the NAF document"""
        dtd = load_dtd(NAF_VERSION_TO_DTD[self.version])
        success = dtd.validate(self.getroot())
        if not success:
            logging.error("DTD error log:")
            for error in dtd.error_log.filter_from_errors():
                logging.error(str(error))
            return success
        return success

    def remove_layer_elements(self, layer: str = None):
        """Remove all elements in layer"""
        layer = self.find(layer)
        for items in layer:
            layer.remove(items)

    def get_attributes(self, data, namespace=None, exclude=list()):
        """ """
        if not isinstance(data, dict):
            data = data._asdict()
        for key, value in dict(data).items():
            if value is None:
                del data[key]
            if isinstance(value, datetime.datetime):
                data[key] = value.strftime("%Y-%m-%dT%H:%M:%SUTC")
            if isinstance(value, float):
                data[key] = str(value)
            if isinstance(value, int):
                data[key] = str(value)
            if isinstance(value, list):
                del data[key]
        if namespace:
            for key, value in dict(data).items():
                qname = etree.QName("{" + namespace + "}" + key, key)
                del data[key]
                data[qname] = value
        for key in dict(data).keys():
            if key in exclude:
                del data[key]
        return data

    def layer(self, layer_tag: str):
        """ """
        layer = self.find(layer_tag)
        if layer is None:
            layer = etree.SubElement(
                self.getroot(), QName(PREFIX_NAF_BASE, layer_tag))
        return layer

    def subelement(
        self,
        element: etree._Element = None,
        tag: str = None,
        data={},
        attributes_to_ignore: list = [],
    ):
        """ """
        if not isinstance(data, dict):
            data = data._asdict()

        for attr in attributes_to_ignore:
            del data[attr]

        subelement = etree.SubElement(
            element,
            QName(PREFIX_NAF_BASE, tag),
            self.get_attributes(data),
        )

        return subelement

    def add_nafHeader(self):
        """ """
        self.layer(NAF_HEADER)

    def add_filedesc_element(self, data: dict):
        """
        FILEDESC ELEMENT
            <fileDesc> is an empty element containing information about the
              computer document itself. It has the following attributes:

              - title: the title of the document (optional).
              - author: the author of the document (optional).
              - creationtime: when the document was created. In ISO 8601. (optional)
              - filename: the original file name (optional).
              - filetype: the original format (PDF, HTML, DOC, etc) (optional).
              - pages: number of pages of the original document (optional).

        ELEMENT fileDesc EMPTY

        ATTLIST fileDesc
            title CDATA #IMPLIED
            author CDATA #IMPLIED
            creationtime CDATA #IMPLIED
            filename CDATA #IMPLIED
            filetype CDATA #IMPLIED
            pages CDATA #IMPLIED
        """
        self.subelement(
            element=self.find(NAF_HEADER),
            tag=FILEDESC_ELEMENT_TAG,
            data=data
        )

    def add_public_element(self, data: dict):
        """
        PUBLIC ELEMENT
            <public> is an empty element which stores public information about
            the document, such as its URI. It has the following attributes:

            - publicId: a public identifier (for instance, the number inserted by the capture server) (optional).
            - uri: a public URI of the document (optional).

        ELEMENT public EMPTY

        ATTLIST public
            publicId CDATA #IMPLIED
            uri CDATA #IMPLIED

        Difference to NAF: here all attributes are mapped to the Dublic Core
        """
        self.subelement(
            element=self.find(NAF_HEADER),
            tag=PUBLIC_ELEMENT_TAG,
            data=self.get_attributes(data, "http://purl.org/dc/elements/1.1/"),
        )

    def add_processor_element(self, layer: str, data: ProcessorElement):
        """
        LINGUISTICPROCESSORS ELEMENT
            <linguisticProcessors> elements store the information about which linguistic processors
            produced the NAF document. There can be several <linguisticProcessors> elements, one
              per NAF layer. NAF layers correspond to the top-level elements of the
              documents, such as "text", "terms", "deps" etc.

        ELEMENT linguisticProcessors (lp)+

        ATTLIST linguisticProcessors
            layer CDATA #REQUIRED

        LP ELEMENT
             <lp> elements describe one specific linguistic processor. <lp> elements
                 have the following attributes:

                 - name: the name of the processor
                 - version: processor's version
                 - timestamp: a timestamp, denoting the date/time at which the processor was
                 launched. The timestamp follows the XML Schema xs:dateTime type (See
                 http://www.w3.org/TR/xmlschema-2/#isoformats). In summary, the date is
                 specified following the form "YYYY-MM-DDThh:mm:ss" (all fields
                 required). To specify a time zone, you can either enter a dateTime in UTC
                 time by adding a "Z" behind the time ("2002-05-30T09:00:00Z") or you can
                 specify an offset from the UTC time by adding a positive or negative time
                 behind the time ("2002-05-30T09:00:00+06:00").
                 - beginTimestamp (optional): a timestamp, denoting the date/time at
                 which the processor started the process. It follows the XML Schema
                 xs:dateTime format.
                 - endTimestamp (optional): a timestamp, denoting the date/time at
                 which the processor ended the process. It follows the XML Schema
                 xs:dateTime format.

        ELEMENT lp EMPTY

        ATTLIST lp
            name CDATA #REQUIRED
            version CDATA #REQUIRED
            timestamp CDATA #IMPLIED
            beginTimestamp CDATA #IMPLIED
            endTimestamp CDATA #IMPLIED
            hostname CDATA #IMPLIED
        """
        proc = self.subelement(
            element=self.find(NAF_HEADER),
            tag=LINGUISTIC_LAYER_TAG,
            data={"layer": layer},
        )
        self.subelement(element=proc, tag=LINGUISTIC_OCCURRENCE_TAG, data=data)

    def add_wf_element(self, data: WordformElement, cdata: bool):
        """
        WORDFORM ELEMENT
            <wf> elements describe and contain all word foorms generated after the tokenization step
              <wf> elements have the following attributes:
                - id: the id of the word form (REQUIRED and UNIQUE)
                - sent: sentence id of the word form (optional)
                - para: paragraph id of the word form (optional)
                - page: page id of the word form (optional)
                - offset: the offset (in characters) of the word form (optional)
                - length: the length (in characters) of the word form (optional)
                - xpath: in case of source xml files, the xpath expression identifying the original word form (optional)

        ELEMENT wf (#PCDATA|subtoken)*

        ATTLIST wf
            id ID #REQUIRED
            sent CDATA #IMPLIED
            para CDATA #IMPLIED
            page CDATA #IMPLIED
            offset CDATA #REQUIRED
            length CDATA #REQUIRED
            xpath CDATA #IMPLIED
        """
        wf = self.subelement(
            element=self.layer(TEXT_LAYER_TAG),
            tag=TEXT_OCCURRENCE_TAG,
            data=data,
            attributes_to_ignore=["text"],
        )

        wf.text = (
            etree.CDATA(
                data.text if "]]>" not in data.text else " " * len(data.text))
            if cdata
            else data.text
        )

    def add_raw_text_element(self, data: RawElement):
        """
        RAW ELEMENT

        ELEMENT raw (#PCDATA)
        """
        self.layer(RAW_LAYER_TAG).text = data.text

    def add_dependency_element(self, data: DependencyRelation, comments: bool):
        """
        DEPS ELEMENT

        ELEMENT deps (dep)+

        DEP ELEMENT
            The <dep> elements have the following attributes:
            -   from: term id of the source element (REQUIRED)
            -   to: term id of the target element (REQUIRED)
            -   rfunc: relational function.(REQUIRED)
            -       case: declension case (optional)

        ELEMENT dep EMPTY
        ATTLIST dep
            from IDREF #REQUIRED
            to IDREF #REQUIRED
            rfunc CDATA #REQUIRED
            case CDATA #IMPLIED
        """
        layer = self.layer(DEPS_LAYER_TAG)

        if comments:
            layer.append(etree.Comment(data.comment))

        self.subelement(
            element=layer,
            tag=DEP_OCCURRENCE_TAG,
            data=data,
            attributes_to_ignore=["comment"],
        )

    def add_entity_element(self, data: EntityElement, comments: bool):
        """
        ENTITY ELEMENT
            A named entity element has the following attributes:
            -   id: the id for the named entity (REQUIRED)
            -   type:  type of the named entity. (IMPLIED) Currently, 8 values are possible:
            -   Person
            -   Organization
            -   Location
            -   Date
            -   Time
            -   Money
            -   Percent
            -   Misc

        ELEMENT entity (span|externalReferences)+

        ATTLIST entity
            id ID #REQUIRED
            type CDATA #IMPLIED
            status CDATA #IMPLIED
            source CDATA #IMPLIED
        """
        element = self.subelement(element=self.layer(ENTITIES_LAYER_TAG), tag=ENTITY_OCCURRENCE_TAG, data=data)

        if data.span != []:
            self.add_span_element(element=element, data=data, comments=comments)

        if data.ext_refs != []:
            self.add_external_reference_element(element=element, ext_refs=data.ext_refs)

    def add_term_element(
        self, data: TermElement, layer_to_attributes_to_ignore: dict, comments: bool
    ):
        """
        TERM ELEMENT
            attributes of term elements
            id: unique identifier (REQUIRED AND UNIQUE)
            type: type of the term. (IMPLIED) Currently, 3 values are possible:
            open: open category term
            close: close category term
            lemma: lemma of the term (IMPLIED).
            pos: part of speech. (IMPLIED)
            Users are encourage to provide URIs to part of speech values to dereference these them.
            more complex pos attributes may be formed by concatenating values separated
            by a dot ".".
            morphofeat: morphosyntactic feature encoded as a single attribute.
            case: declension case of the term (optional).
            head: if the term is a compound, the id of the head component (optional).
            component_of: if the term is part of multiword, i.e., referenced by a multiwords/mw element
            than this attribute can be used to make reference to the multiword.
            compound_type: endocentric or exocentric

        ELEMENT term (sentiment?|span|externalReferences|component)+
        ATTLIST term
            id ID #REQUIRED
            type CDATA #IMPLIED
            lemma CDATA #IMPLIED
            pos CDATA #IMPLIED
            morphofeat CDATA #IMPLIED
            netype CDATA #IMPLIED
            case CDATA #IMPLIED
            head CDATA #IMPLIED
            component_of IDREF #IMPLIED
            compound_type CDATA #IMPLIED
        """
        element = self.subelement(
            element=self.layer(TERMS_LAYER_TAG),
            tag=TERM_OCCURRENCE_TAG,
            data=data,
            attributes_to_ignore=layer_to_attributes_to_ignore.get("terms", list()),
        )

        if data.span != []:
            self.add_span_element(element=element, data=data, comments=comments)

        if data.ext_refs != []:
            self.add_external_reference_element(element=element, ext_refs=data.ext_refs)

    def add_chunk_element(self, data: ChunkElement, comments: bool):
        """
        CHUNKS ELEMENT

        ELEMENT chunks (chunk)+

        CHUNK ELEMENT
            The <chunk> elements have the following attributes:
            -   id: unique identifier (REQUIRED)
            -   head: the chunk head’s term id  (REQUIRED)
            -   phrase: type of the phrase (REQUIRED)
            -   case: declension case (optional)

        ELEMENT chunk (span)+

        ATTLIST chunk
            id ID #REQUIRED
            head IDREF #REQUIRED
            phrase CDATA #REQUIRED
            case CDATA #IMPLIED
        """
        element = self.subelement(element=self.layer(CHUNKS_LAYER_TAG), tag=CHUNK_OCCURRENCE_TAG, data=data)

        if data.span != []:
            self.add_span_element(element=element, data=data, comments=comments)

    def add_span_element(self, element, data, comments=False):
        """
        SPAN ELEMENT

        ELEMENT span (target)+

        ATTLIST span
            primary CDATA #IMPLIED
            status CDATA #IMPLIED
        """
        if not isinstance(data, dict):
            data = data._asdict()

        if (self.version is not None) and (self.version == "v3"):
            references = self.subelement(element=element, tag="references")
            span = self.subelement(element=references, tag=SPAN_OCCURRENCE_TAG)
        else:
            span = self.subelement(element=element, tag=SPAN_OCCURRENCE_TAG)
        if comments:
            comment = " ".join(data["comment"])
            comment = prepare_comment_text(comment)
            span.append(etree.Comment(comment))

        for target in data["span"]:
            self.subelement(element=span, tag=TARGET_OCCURRENCE_TAG, data={"id": target})

    def add_external_reference_element(self, element, ext_refs: list):
        """
        EXTERNALREFERENCES ELEMENT
            The <externalReferences> element is used to associate terms to
            external resources, such as elements of a Knowledge base, an ontology,
            etc. It consists of several <externalRef> elements, one per
            association.

        ELEMENT externalReferences (externalRef)+

        EXTERNALREF ELEMENT
            <externalRef> elements have the following attributes:- resource: indicates the identifier of the resource
            referred to.
            - reference: code of the referred element. If the element is a
            synset of some version of WordNet, it follows the pattern:
            [a-z]{3}-[0-9]{2}-[0-9]+-[nvars]
            which is a string composed by four fields separated by a dash.
            The four fields are the following:
            - Language code (three characters).
            - WordNet version (two digits).
            - Synset identifier composed by digits.
            - POS character:
            n noun
            v verb
            a adjective
            r adverb
            examples of valid patterns are: ``ENG-20-12345678-n'',
            ``SPA-16-017403-v'', etc.
            - confidence: a floating number between 0 and 1. Indicates the confidence weight of the association

        ELEMENT externalRef (sentiment|externalRef)*

        ATTLIST externalRef
            resource CDATA #IMPLIED
            reference CDATA #REQUIRED
            reftype CDATA #IMPLIED
            status CDATA #IMPLIED
            source CDATA #IMPLIED
            confidence CDATA #IMPLIED
            timestamp CDATA #IMPLIED
        """
        if not isinstance(ext_refs, list):
            logging.info(
                "ext_refs should be a list of dictionaries (can be empty)")

        ext_refs_el = self.subelement(
            element=element, tag=EXT_REFS_OCCURRENCE_TAG)
        for ext_ref in ext_refs:
            ext_ref_el = self.subelement(
                element=ext_refs_el,
                tag=EXT_REF_OCCURRENCE_TAG,
                data={"reference": ext_ref["reference"]},
            )
            for optional_attr in ["resource", "source", "timestamp"]:
                if optional_attr in ext_ref:
                    ext_ref_el.set(optional_attr, ext_ref[optional_attr])

    def add_multiword_element(self, data: MultiwordElement):
        """
        MULTIWORDS ELEMENT

        ELEMENT multiwords (mw)+

        MW ELEMENT
            attributes of mw elements
            id: unique identifier (REQUIRED AND UNIQUE)
            lemma: lemma of the term (IMPLIED).
            pos: part of speech. (IMPLIED)
            morphofeat: morphosyntactic feature encoded as a single attribute. (IMPLIED)
            case: declension case (IMPLIED)
            status: manual | system | deprecated
            type: phrasal, idiom

        ELEMENT mw (component|externalReferences)+

        ATTLIST mw
            id ID #REQUIRED
            lemma CDATA #IMPLIED
            pos CDATA #IMPLIED
            morphofeat CDATA #IMPLIED
            case CDATA #IMPLIED
            status CDATA #IMPLIED
            type CDATA #REQUIRED
        """
        mw = self.subelement(
            element=self.layer(MULTIWORDS_LAYER_TAG),
            tag=MULTIWORD_OCCURRENCE_TAG,
            data=data,
        )
        for component in data.components:
            com = self.subelement(
                element=mw, tag=COMPONENT_OCCURRENCE_TAG, data=component
            )
            self.add_span_element(element=com, data=component)
