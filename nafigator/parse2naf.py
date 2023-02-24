# coding: utf-8

"""Main module."""

from datetime import datetime
import io
import logging
from lxml import etree
import lxml.html
from nifigator import align_stanza_dict_offsets
from nifigator.utils import tokenizer
from socket import getfqdn
import os
import sys
from typing import Union

from .nafdocument import NafDocument
from .linguisticprocessor import stanzaProcessor
from .linguisticprocessor import spacyProcessor
from .preprocessprocessor import convert_docx
from .ocrprocessor import convert_ocr_pdf

from .const import ProcessorElement
from .const import Entity
from .const import WordformElement
from .const import TermElement
from .const import EntityElement
from .const import DependencyRelation
from .const import ChunkElement
from .const import RawElement
from .const import MultiwordElement
from .const import ComponentElement
from .utils import normalize_token_orth
from .utils import remove_illegal_chars
from .utils import prepare_comment_text

FORMATS_LAYER_TAG = "formats"


def generate_naf(
    input: Union[str, NafDocument] = None,
    stream: io.BytesIO = None,
    engine: str = None,
    language: str = None,
    naf_version: str = None,
    dtd_validation: bool = False,
    text: str = None,
    params: dict = {},
    nlp=None,
):
    """Parse input file, generate and return NAF xml tree"""
    if input is None:
        logging.error("input is none")
        return None
    if isinstance(input, str) and not os.path.isfile(input) and stream == None:
        logging.error("no or non-existing input specified")
        return None
    if engine is None:
        logging.error("no engine specified")
        return None
    if (language is None) and ("language_detector" not in params.keys()):
        logging.error("no language or language detector specified")
        return None
    if naf_version is None:
        logging.error("no naf version specified")
        return None
    if engine.lower() == "stanza" and "stanza" not in sys.modules:
        logging.error("stanza not installed")
        return None
    if engine.lower() == "spacy" and "spacy" not in sys.modules:
        logging.error("SpaCy not installed")
        return None

    params = create_params(
        input=input,
        stream=stream,
        engine=engine,
        language=language,
        naf_version=naf_version,
        dtd_validation=dtd_validation,
        params=params,
        nlp=nlp,
    )

    if isinstance(input, NafDocument):
        params["tree"] = input
    else:
        params["tree"] = NafDocument()
        params["tree"].generate(params)

    if params["preprocess_layers"] != []:
        process_preprocess_steps(params, text)

    if params["linguistic_layers"] != []:
        process_linguistic_steps(params, text)
        evaluate_naf(params, text)

    return params["tree"]


def create_params(
    input: str = None,
    stream: io.BytesIO = None,
    engine: str = None,
    language: str = None,
    naf_version: str = None,
    dtd_validation: bool = False,
    params: dict = {},
    nlp=None,
):
    """Return params dictionary with all params used in the parse process"""
    params["naf_version"] = naf_version
    params["dtd_validation"] = bool(dtd_validation)
    params["language"] = language
    params["engine_name"] = engine
    params["nlp"] = nlp

    if isinstance(input, str):
        if "fileDesc" not in params.keys():
            params["fileDesc"] = dict()
        params["fileDesc"]["creationtime"] = datetime.now()
        params["fileDesc"]["filename"] = input

        if "public" not in params.keys():
            params["public"] = dict()
            params["public"]["uri"] = input
        elif "uri" not in params["public"].keys():
            params["public"]["uri"] = input
        if stream is not None:
            if not isinstance(stream, io.BytesIO):
                stream = io.BytesIO(stream)
            params['stream'] = stream

        if os.path.splitext(input)[1].lower() == ".txt":
            params["fileDesc"]["filetype"] = "text/plain"
            params["public"]["format"] = "text/plain"
        elif os.path.splitext(input)[1].lower() == ".html":
            params["fileDesc"]["filetype"] = "text/html"
            params["public"]["format"] = "text/html"
        elif os.path.splitext(input)[1].lower() == ".pdf":
            params["fileDesc"]["filetype"] = "application/pdf"
            params["public"]["format"] = "application/pdf"
        elif os.path.splitext(input)[1].lower() == ".docx":
            params["fileDesc"][
                "filetype"
            ] = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            params["public"][
                "format"
            ] = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

    if isinstance(input, bytes):
        if "fileDesc" not in params.keys():
            params["fileDesc"] = dict()
        params["fileDesc"]["creationtime"] = datetime.now()
        params["fileDesc"]["filename"] = input

        if "public" not in params.keys():
            params["public"] = dict()
            params["public"]["uri"] = input
        else:
            if "uri" not in params["public"].keys():
                params["public"]["uri"] = input
        if stream is not None:
            params['stream'] = stream

        if os.path.splitext(input)[1].lower() == ".txt":
            params["fileDesc"]["filetype"] = "text/plain"
            params["public"]["format"] = "text/plain"
        elif os.path.splitext(input)[1].lower() == ".html":
            params["fileDesc"]["filetype"] = "text/html"
            params["public"]["format"] = "text/html"
        elif os.path.splitext(input)[1].lower() == ".pdf":
            params["fileDesc"]["filetype"] = "application/pdf"
            params["public"]["format"] = "application/pdf"
        elif os.path.splitext(input)[1].lower() == ".docx":
            params["fileDesc"][
                "filetype"
            ] = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            params["public"][
                "format"
            ] = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

    # set default linguistic parameters
    if "linguistic_layers" not in params.keys():
        params["linguistic_layers"] = [
            "text",
            "terms",
            "entities",
            "deps",
            "raw",
            "multiwords",
        ]
    if "preprocess_layers" not in params.keys():
        params["preprocess_layers"] = ["formats"]
    if params.get("cdata", None) is None:
        params["cdata"] = True
    if params.get("map_udpos2olia", None) is None:
        params["map_udpos2olia"] = False
    if params.get("layer_to_attributes_to_ignore", None) is None:
        params["layer_to_attributes_to_ignore"] = {"terms": {}}
    if params.get("replace_hidden_characters", None) is None:
        params["replace_hidden_characters"] = True
    if params.get("comments", None) is None:
        params["comments"] = True
    if params.get("apply_ocr", None) is None:
        params["apply_ocr"] = False
    if params.get("textline_separator") is None:
        params["textline_separator"] = " "

    return params


def evaluate_naf(params: dict, text: str):
    """Perform alignment between raw layer, document text and text layer in the NAF xml tree"""
    # verify alignment between raw layer and document text
    doc_text = params["engine"].document_text(params["doc"])
    raw = params["tree"].raw
    if len(raw) != len(doc_text):
        logging.error(f"raw length ({len(raw)}) != doc length ({len(doc_text)})")
    # verify alignment between raw layer and text
    text_to_use = text
    if len(raw) != len(text_to_use):
        logging.error(f"raw length ({len(raw)}) != text to use ({len(text_to_use)})")
    # verify alignment between raw layer and text layer
    for wf in params["tree"].text:
        start = int(wf.get("offset"))
        end = start + int(wf.get("length"))
        token = raw[start:end]
        if wf.get("text", None) != token:
            logging.error(
                f"mismatch in alignment of wf element [{wf.get('text')}]"
                f"({wf.get('id')}) with raw layer text [{token}]"
                f"(expected length {wf.get('length')})"
            )
    # validate naf tree
    if params["dtd_validation"]:
        params["tree"].validate()


def process_preprocess_steps(params: dict, text: str):
    """
    Perform preprocessor steps to generate text as input for linguistic layers
    Added to params: beginTimestamp_preprocess, endTimestamp_preprocess
    Note text param to be removed
    """
    params["beginTimestamp_preprocess"] = datetime.now()
    input = params["fileDesc"]["filename"]
    if input[-3:].lower() == "txt":
        stream = params.get("stream", None)
        if stream is not None:
            text = stream.read()
        else:
            with open(input, encoding="utf8") as f:
                text = f.read()
        params["text"] = text
    elif input[-4:].lower() == "html":
        stream = params.get("stream", None)
        if stream is not None:
            text = stream.read()
        else:
            with open(input, encoding="utf8") as f:
                text = f.read()
        utf8_parser = lxml.html.HTMLParser(encoding="utf-8")
        doc = lxml.html.document_fromstring(bytes(text, encoding='utf8'), parser=utf8_parser)
        params["text"] = doc.text_content()
    elif input[-4:].lower() == "docx":
        convert_docx(input, format="xml", params=params)
        convert_docx(input, format="text", params=params)
    elif input[-3:].lower() == "pdf":
        if not params["apply_ocr"]:
            params["text"] = text
        else:
            params["text"] = convert_ocr_pdf(input, format="text", params=params)

    params["endTimestamp_preprocess"] = datetime.now()


def process_linguistic_steps(params: dict, text: str):
    """Perform linguistic steps to generate linguistics layers"""
    engine_name = params["engine_name"]
    nlp = params["nlp"]

    # determine language for nlp processor
    if params["language"] is not None:
        language = params["language"]
    else:
        language = params["language_detector"].detect(text)
        params["tree"].set_language(language)
        params["language"] = language

    # create nlp processor
    if engine_name.lower() == "stanza":
        # check if installed
        params["engine"] = stanzaProcessor(nlp, language)
    elif engine_name.lower() == "spacy":
        # check if installed
        params["engine"] = spacyProcessor(nlp, language)
    else:
        logging.error("unknown engine")
        return None

    # execute nlp processor pipeline
    params["beginTimestamp"] = datetime.now()
    tokenized_text = tokenizer(text)

    # correction for bug in stanza
    if tokenized_text != []:
        if tokenized_text[-1][-1] == "":
            tokenized_text[-1] = tokenized_text[-1][:-1]

    # extract the text from the tokenized data
    sentences_text = [[word['text'] for word in sentence] for sentence in tokenized_text]
    params["doc"] = params["engine"].nlp(sentences_text)
    params["endTimestamp"] = datetime.now()

    # derive naf layers from nlp output
    process_linguistic_layers(params, text, tokenized_text)


def process_linguistic_layers(params: dict, text: str, tokenized_text: list):
    """Perform linguistic layers"""
    layers = params["linguistic_layers"]

    if "entities" in layers:
        add_entities_layer(params)

    if "text" in layers:
        add_text_layer(params, tokenized_text)

    if "terms" in layers:
        add_terms_layer(params)

    if "deps" in layers:
        add_deps_layer(params)

    if "multiwords" in layers:
        add_multiwords_layer(params)

    if "chunks" in layers:
        add_chunks_layer(params)

    if "raw" in layers:
        add_raw_layer(params, text)


def entities_generator(doc, params: dict):
    """Return entities in doc as a generator"""
    engine = params["engine"]
    for ent in engine.document_entities(doc):
        yield Entity(
            start=engine.entity_span_start(ent),
            end=engine.entity_span_end(ent),
            type=engine.entity_type(ent),
        )


def chunks_for_doc(doc, params: dict):
    """Return chunks in doc as a generator"""
    for chunk in params["engine"].document_noun_chunks(doc):
        if chunk.root.head.pos_ == "ADP":
            span = doc[chunk.start - 1: chunk.end]
            yield (span, "PP")
        yield (chunk, "NP")


def chunk_tuples_for_doc(doc, params: dict):
    """Return chunk tuples as a generator"""
    for i, (chunk, phrase) in enumerate(chunks_for_doc(doc, params)):
        comment = remove_illegal_chars(chunk.orth_.replace("\n", " "))
        yield ChunkElement(
            id="c" + str(i),
            head="t" + str(chunk.root.i),
            phrase=phrase,
            case=None,
            span=["t" + str(tok.i) for tok in chunk],
            comment=comment,
        )


def dependencies_to_add(sentence, token, total_tokens: int, params: dict):
    """Generate list of dependencies to add to deps layer"""
    engine = params["engine"]
    deps = list()
    cor = engine.offset_token_index()

    while engine.token_head_index(sentence, token) != engine.token_index(token):
        from_term = "t" + str(
            engine.token_head_index(sentence, token) + total_tokens + cor
        )
        to_term = "t" + str(engine.token_index(token) + total_tokens + cor)
        rfunc = engine.token_dependency(token)
        from_orth = engine.token_orth(engine.token_head(sentence, token))
        to_orth = engine.token_orth(token)

        comment = rfunc + "(" + from_orth + "," + to_orth + ")"
        comment = prepare_comment_text(comment)

        dep_data = DependencyRelation(
            from_term=from_term,
            to_term=to_term,
            rfunc=rfunc,
            case=None,
            comment=comment,
        )
        deps.append(dep_data)
        token = engine.token_head(sentence, token)
    return deps


def add_entities_layer(params: dict):
    """Generate and add all entities in document to entities layer"""
    lp = ProcessorElement(
        name="entities",
        version=params["engine"].model_version,
        model=params["engine"].processor("entities").get("model", ""),
        timestamp=None,
        beginTimestamp=params["beginTimestamp"],
        endTimestamp=params["endTimestamp"],
        hostname=getfqdn(),
    )

    params["tree"].add_processor_element("entities", lp)

    doc = params["doc"]
    engine = params["engine"]

    current_entity = list()  # Use a list for multiword entities.
    current_entity_orth = list()  # id.

    current_token: int = 1  # Keep track of the token number.
    term_number: int = 1  # Keep track of the term number.
    entity_number: int = 1  # Keep track of the entity number.
    total_tokens: int = 0

    parsing_entity: bool = False

    for sentence_number, sentence in enumerate(engine.document_sentences(doc), start=1):

        entity_gen = entities_generator(sentence, params)
        try:
            next_entity = next(entity_gen)
        except StopIteration:
            next_entity = Entity(start=None, end=None, type=None)

        for token_number, token in enumerate(
            engine.sentence_tokens(sentence), start=current_token
        ):
            # Do we need a state change?

            if token_number == next_entity.start:
                parsing_entity = True

            tid = "t" + str(term_number)
            if parsing_entity:
                current_entity.append(tid)
                current_entity_orth.append(
                    normalize_token_orth(engine.token_orth(token))
                )

            # Move to the next term
            term_number += 1

            if parsing_entity and token_number == next_entity.end:
                # Create new entity ID.
                entity_id = "e" + str(entity_number)
                # Create Entity data:
                entity_data = EntityElement(
                    id=entity_id,
                    type=next_entity.type,
                    status=None,
                    source=None,
                    span=current_entity,
                    ext_refs=list(),
                    comment=current_entity_orth,
                )

                params["tree"].add_entity_element(
                    entity_data, params["naf_version"], params["language"]
                )

                entity_number += 1
                current_entity = list()
                current_entity_orth = list()

                # Move to the next entity
                parsing_entity = False
                try:
                    next_entity = next(entity_gen)
                except StopIteration:
                    next_entity = Entity(start=None, end=None, type=None)

        if engine.token_reset() is False:
            current_token = token_number + 1
            total_tokens = 0
        else:
            current_token = 1
            total_tokens += token_number

    return None


def add_text_layer(params: dict, tokenized_text: list):
    """Generate and add all words in document to text layer"""
    lp = ProcessorElement(
        name="text",
        version=params["engine"].model_version,
        model=params["engine"].processor("text").get("model", ""),
        timestamp=None,
        beginTimestamp=params["beginTimestamp"],
        endTimestamp=params["endTimestamp"],
        hostname=getfqdn(),
    )

    params["tree"].add_processor_element("text", lp)

    # align the stanza output with the tokenized text
    doc = align_stanza_dict_offsets(params["doc"].to_dict(), tokenized_text)

    wf_count_prev_sent = 0
    idx_w = 0
    for idx_s, s in enumerate(doc, start=1):
        wf_count_prev_sent += idx_w
        for idx_w, wf in enumerate(s, start=1):
            wf_id = "w" + str(wf_count_prev_sent + idx_w)
            wf_data = WordformElement(
                id=wf_id,
                sent=str(idx_s),
                para=None,  # TODO nog te fixen -> nu geen paragraphs layer
                page=None,  # TODO nog te fixen -> page offset??
                offset=str(wf["start_char"]),
                length=str(len(wf["text"])),
                xpath=None,
                text=wf["text"],
            )

            params["tree"].add_wf_element(wf_data, params["cdata"])

    return None


def add_terms_layer(params: dict):
    """Generate and add all terms in document to terms layer"""
    lp = ProcessorElement(
        name="terms",
        version=params["engine"].model_version,
        model=params["engine"].processor("terms").get("model", ""),
        timestamp=None,
        beginTimestamp=params["beginTimestamp"],
        endTimestamp=params["endTimestamp"],
        hostname=getfqdn(),
    )

    params["tree"].add_processor_element("terms", lp)

    doc = params["doc"]
    engine = params["engine"]

    current_term = list()  # Use a list for multiword expressions.
    current_term_orth = list()  # id.

    current_token: int = 1  # Keep track of the token number.
    term_number: int = 1  # Keep track of the term number.
    total_tokens: int = 0

    for sentence_number, sentence in enumerate(engine.document_sentences(doc), start=1):

        for token_number, token in enumerate(
            engine.sentence_tokens(sentence), start=current_token
        ):

            wid = "w" + str(token_number + total_tokens)
            tid = "t" + str(term_number)

            current_term.append(wid)
            current_term_orth.append(normalize_token_orth(engine.token_orth(token)))

            # Create TermElement data:
            token_pos = engine.token_pos(token)
            # :param bool map_udpos2naf_pos: if True, we use "udpos2nafpos_info"
            # to map the Universal Dependencies pos (https://universaldependencies.org/u/pos/)
            # to the NAF pos tagset
            if params["map_udpos2olia"]:
                if token_pos in udpos2olia.keys():
                    pos_type = udpos2olia[token_pos]["class"]
                    token_pos = udpos2olia[token_pos]["olia"]
                else:
                    logging.info("unknown token pos: " + str(token_pos))
                    pos_type = "open"
                    token_pos = "unknown"
            else:
                pos_type = "open"

            term_data = TermElement(
                id=tid,
                type=pos_type,
                lemma=remove_illegal_chars(engine.token_lemma(token)),
                pos=token_pos,
                morphofeat=engine.token_tag(token),
                netype=None,
                case=None,
                head=None,
                component_of=None,
                compound_type=None,
                span=current_term,
                ext_refs=list(),
                comment=current_term_orth,
            )

            params["tree"].add_term_element(
                term_data, params["layer_to_attributes_to_ignore"], params["comments"]
            )

            # Move to the next term
            term_number += 1
            current_term = list()
            current_term_orth = list()

        # At the end of the sentence,
        # add all the dependencies to the XML structure.
        if engine.token_reset() is False:
            current_token = token_number + 1
            total_tokens = 0
        else:
            current_token = 1
            total_tokens += token_number

    return None


def add_deps_layer(params: dict):
    """Generate and add all dependencies in document to deps layer"""
    lp = ProcessorElement(
        name="deps",
        version=params["engine"].model_version,
        model=params["engine"].processor("deps").get("model", ""),
        timestamp=None,
        beginTimestamp=params["beginTimestamp"],
        endTimestamp=params["endTimestamp"],
        hostname=getfqdn(),
    )

    params["tree"].add_processor_element("deps", lp)

    engine = params["engine"]

    current_token: int = 1
    total_tokens: int = 0

    for sent in engine.document_sentences(params["doc"]):

        dependencies_for_sentence = list()

        for token_number, token in enumerate(
            engine.sentence_tokens(sent), start=current_token
        ):
            for dep_data in dependencies_to_add(sent, token, total_tokens, params):
                if dep_data not in dependencies_for_sentence:
                    dependencies_for_sentence.append(dep_data)

        for dep_data in dependencies_for_sentence:
            params["tree"].add_dependency_element(dep_data, params["comments"])

        if engine.token_reset() is False:
            current_token = token_number + 1
            total_tokens = 0
        else:
            current_token = 1
            total_tokens += token_number

    return None


def get_next_mw_id(params):
    """Return multiword id for new multiword"""
    layer = params["tree"].find("multiwords")
    if layer is None:
        layer = etree.SubElement(params["tree"].getroot(), "multiwords")
    mw_ids = [int(mw_el.get("id")[2:]) for mw_el in layer.xpath("mw")]
    if mw_ids:
        next_mw_id = max(mw_ids) + 1
    else:
        next_mw_id = 1
    return f"mw{next_mw_id}"


def create_separable_verb_lemma(verb, particle, language):
    """return lemma (particle plus verb) for nl and en"""
    if language == "nl":
        lemma = particle + verb
    if language == "en":
        lemma = f"{verb}_{particle}"
    else:
        lemma = f"{verb}_{particle}"
    return lemma


def add_multiwords_layer(params: dict):
    """Generate and add all multiwords in document to multiwords layer"""
    lp = ProcessorElement(
        name="multiwords",
        version=params["engine"].model_version,
        model=params["engine"].processor("multiwords").get("model", ""),
        timestamp=None,
        beginTimestamp=params["beginTimestamp"],
        endTimestamp=params["endTimestamp"],
        hostname=getfqdn(),
    )

    params["tree"].add_processor_element("multiwords", lp)

    engine = params["engine"]

    if params["naf_version"] == "v3":
        logging.info("add_multi_words function only applies to naf version 4")

    supported_languages = {"nl", "en"}
    if params["language"] not in supported_languages:
        logging.info(
            f"add_multi_words function only implemented for "
            f"{supported_languages}, not for supplied {params['language']}"
        )

    tid_to_term = {
        term.get("id"): term for term in params["tree"].findall("terms/term")
    }

    for dep in params["tree"].deps:

        if dep.get("rfunc") == "compound:prt":

            next_mw_id = get_next_mw_id(params)

            idverb = dep.get("from_term")
            idparticle = dep.get("to_term")

            verb_term_el = tid_to_term[idverb]
            verb = verb_term_el.attrib.get("lemma")
            verb_term_el.set("component_of", next_mw_id)

            particle_term_el = tid_to_term[idparticle]
            particle = particle_term_el.attrib.get("lemma")
            particle_term_el.set("component_of", next_mw_id)

            separable_verb_lemma = create_separable_verb_lemma(
                verb, particle, params["language"]
            )
            multiword_data = MultiwordElement(
                id=next_mw_id,
                lemma=separable_verb_lemma,
                pos="VERB",
                morphofeat=None,
                case=None,
                status=None,
                type="phrasal",
                components=[],
            )

            components = [
                (f"{next_mw_id}.c1", idverb),
                (f"{next_mw_id}.c2", idparticle),
            ]

            for c_id, t_id in components:
                component_data = ComponentElement(
                    id=c_id,
                    type=None,
                    lemma=None,
                    pos=None,
                    morphofeat=None,
                    netype=None,
                    case=None,
                    head=None,
                    span=[t_id],
                )
                multiword_data.components.append(component_data)

            params["tree"].add_multiword_element(multiword_data)

            # component = etree.SubElement(
            #     mw_element, "component", attrib={"id": c_id}
            # )
            # span = etree.SubElement(component, "span")
            # etree.SubElement(span, "target", attrib={"id": t_id})

    # params["tree"].add_multi_words(params["naf_version"], params["language"])


def add_raw_layer(params: dict, text: str):
    """Generate and add raw text in document to raw layer"""
    lp = ProcessorElement(
        name="raw",
        version=params["engine"].model_version,
        model=params["engine"].processor("raw").get("model", ""),
        timestamp=None,
        beginTimestamp=params["beginTimestamp"],
        endTimestamp=params["endTimestamp"],
        hostname=getfqdn(),
    )

    params["tree"].add_processor_element("raw", lp)
    raw_data = RawElement(text=text)
    params["tree"].add_raw_text_element(raw_data)


def add_chunks_layer(params: dict):
    """Generate and add all chunks in document to chunks layer"""
    lp = ProcessorElement(
        name="chunks",
        version=params["engine"].model_version,
        model=params["engine"].processor("chunks").get("model", ""),
        timestamp=None,
        beginTimestamp=params["beginTimestamp"],
        endTimestamp=params["endTimestamp"],
        hostname=getfqdn(),
    )

    params["tree"].add_processor_element("chunks", lp)

    for chunk_data in chunk_tuples_for_doc(params["doc"], params):
        params["tree"].add_chunk_element(chunk_data, params["comments"])
