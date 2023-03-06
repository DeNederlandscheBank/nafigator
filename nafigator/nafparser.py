# coding: utf-8

"""Main module."""

import sys
import logging
import os
import io
from datetime import datetime
from socket import getfqdn
from lxml import etree
import numpy as np
from typing import Union, Optional
from nifigator.utils import align_stanza_dict_offsets, tokenizer

from .linguisticprocessor import stanzaProcessor, spacyProcessor
from .nafdocument.nafdocument import NafDocument
from .nafdocument.const import (
    TERMS_LAYER_TAG,
    DEPS_LAYER_TAG,
    TEXT_LAYER_TAG,
    ENTITIES_LAYER_TAG,
    CHUNKS_LAYER_TAG,
    MULTIWORDS_LAYER_TAG,
    RAW_LAYER_TAG
)
from .nafdocument.nafelements import (
    ProcessorElement,
    Entity,
    WordformElement,
    TermElement,
    EntityElement,
    DependencyRelation,
    ChunkElement,
    RawElement,
    MultiwordElement,
    ComponentElement
)
from .const import udpos2nafpos_info
from .utils import (
    normalize_token_orth,
    remove_illegal_chars,
    prepare_comment_text,
    path_to_format
)


class NafParser():
    def __init__(
        self,
        engine: str,
        language: str,
        naf_version: str,
        dtd_validation: bool = False,
        nlp=None,
    ) -> None:
        """Parse input file, generate and return NAF xml tree"""
        if engine.lower() == "stanza" and "stanza" not in sys.modules:
            logging.error("stanza not installed")
            return None
        if engine.lower() == "spacy" and "spacy" not in sys.modules:
            logging.error("SpaCy not installed")
            return None

        self.engine = engine
        self.language = language
        self.naf_version = naf_version
        self.dtd_validation = dtd_validation
        self.nlp = nlp

    def generate_naf(
        self,
        input: Union[str, NafDocument],
        stream: io.BytesIO = None,
        text: str = None,
        pdfdoc: object = None,
        params: dict = {}
    ) -> NafDocument:
        if isinstance(input, str) and not os.path.isfile(input) and stream is None:
            logging.error("no or non-existing input specified")
            return None
        if (self.language is None) and ("language_detector" not in params.keys()):
            logging.error("no language or language detector specified")
            return None
        if (text is None) and ((pdfdoc is None) or not hasattr(pdfdoc, "text")):
            logging.error("text or pdfdoc text is not specified")
            return None

        if isinstance(input, NafDocument):
            self.nafdoc = input
        else:
            if isinstance(input, str):
                filedesc_params = params.get("filedesc", {})
                filedesc_params = self.add_filedesc_params(input, filedesc_params)
                public_params = params.get("public", {})
                public_params = self.add_public_params(input, public_params)
                params = self.add_stream_params(params, stream)

            self.nafdoc = NafDocument(
                naf_version=self.naf_version,
                language=self.language,
                filedesc_elem=filedesc_params,
                public_elem=public_params
            )

        self.text = pdfdoc.text if (text is None) else text

        if pdfdoc is not None:
            self.page_offsets = pdfdoc.page_offsets
            self.paragraph_offsets = pdfdoc.paragraph_offsets

        self.params = params
        self.set_default_params(params)

        if NafDocument.layers != []:
            self.process_linguistic_steps(self.params)
            self.evaluate_naf()

        return self.nafdoc

    def set_default_params(self, params: dict) -> None:
        # TODO: are they coming from params? And should they be initialized?
        self.cdata = params.get("cdata", True)
        self.map_udpos2olia = params.get("map_udpos2olia", False)
        self.layer_to_attributes_to_ignore = params.get("layer_to_attributes_to_ignore", {"terms": {}})
        self.replace_hidden_characters = params.get("replace_hidden_characters", True)
        self.comments = params.get("comments", True)
        self.textline_separator = params.get("textline_separator", " ")

    def add_filedesc_params(self, input: str, filedesc_params: dict = {}) -> dict:
        """Return params dictionary with filedesc params"""
        filedesc_params["creationtime"] = datetime.now()
        filedesc_params["filename"] = input
        filedesc_params["filetype"] = path_to_format(input)

        return filedesc_params

    def add_public_params(self, input: str, public_params: dict = {}) -> dict:
        """Return params dictionary with public params"""
        public_params["uri"] = input
        public_params["format"] = path_to_format(input)

        return public_params

    def add_stream_params(self, params: dict, stream: Optional[io.BytesIO]):
        """Return params dictionary with stream params"""
        if stream is not None:
            if not isinstance(stream, io.BytesIO):
                stream = io.BytesIO(stream)
            params["stream"] = stream

        return params

    def evaluate_naf(self):
        """Perform alignment between raw layer, document text and text layer in the NAF xml tree"""
        # verify alignment between raw layer and document text
        doc_length = self.tokenized_text[-1][-1]["end_char"]
        raw = self.nafdoc.raw
        if len(raw) != doc_length:
            logging.error(f"raw length ({len(raw)}) != doc length ({doc_length})")

        # verify alignment between raw layer and text layer
        for wf in self.nafdoc.text:
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
        if self.dtd_validation:
            self.nafdoc.validate()

    def process_linguistic_steps(self, params: dict):
        """Perform linguistic steps to generate linguistics layers"""
        # determine language for nlp processor
        if self.language is not None:
            language = self.language
        else:
            language = params["language_detector"].detect(self.text)
            self.nafdoc.set_language(language)
            self.language = language

        # create nlp processor
        if self.engine.lower() == "stanza":
            # check if installed
            self.engine = stanzaProcessor(self.nlp, language)
        elif self.engine.lower() == "spacy":
            # check if installed
            self.engine = spacyProcessor(self.nlp, language)
        else:
            logging.error("unknown engine")
            return None

        self.begin_timestamp = datetime.now()

        # tokenize text
        tokenized_text = tokenizer(self.text)
        self.tokenized_text = tokenized_text

        # correction for bug in stanza
        if tokenized_text != []:
            if tokenized_text[-1][-1] == "":
                tokenized_text[-1] = tokenized_text[-1][:-1]

        # extract the text from the tokenized data
        sentences_text = [[word['text'] for word in sentence] for sentence in tokenized_text]

        # execute nlp processor pipeline
        self.doc = self.engine.nlp(sentences_text)
        self.end_timestamp = datetime.now()

        # derive naf layers from nlp output
        self.process_linguistic_layers()

    def process_linguistic_layers(self):
        """Perform linguistic layers"""
        layers = self.nafdoc.layers

        if ENTITIES_LAYER_TAG in layers:
            self.add_entities_layer()

        if TEXT_LAYER_TAG in layers:
            self.add_text_layer()

        if TERMS_LAYER_TAG in layers:
            self.add_terms_layer()

        if DEPS_LAYER_TAG in layers:
            self.add_deps_layer()

        if MULTIWORDS_LAYER_TAG in layers:
            self.add_multiwords_layer()

        if CHUNKS_LAYER_TAG in layers:
            self.add_chunks_layer()

        if RAW_LAYER_TAG in layers:
            self.add_raw_layer()

    def entities_generator(self, doc):
        """Return entities in doc as a generator"""
        engine = self.engine
        for ent in engine.document_entities(doc):
            yield Entity(
                start=engine.entity_span_start(ent),
                end=engine.entity_span_end(ent),
                type=engine.entity_type(ent),
            )

    def chunks_for_doc(self, doc):
        """Return chunks in doc as a generator"""
        for chunk in self.engine.document_noun_chunks(doc):
            if chunk.root.head.pos_ == "ADP":
                span = doc[chunk.start - 1: chunk.end]
                yield (span, "PP")
            yield (chunk, "NP")

    def chunk_tuples_for_doc(self, doc):
        """Return chunk tuples as a generator"""
        for i, (chunk, phrase) in enumerate(self.chunks_for_doc(doc)):
            comment = remove_illegal_chars(chunk.orth_.replace("\n", " "))
            yield ChunkElement(
                id="c" + str(i),
                head="t" + str(chunk.root.i),
                phrase=phrase,
                case=None,
                span=["t" + str(tok.i) for tok in chunk],
                comment=comment,
            )

    def dependencies_to_add(self, sentence, token, total_tokens: int):
        """Generate list of dependencies to add to deps layer"""
        engine = self.engine
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

    def add_layer(self, name: str):
        lp = ProcessorElement(
            name=name,
            version=self.engine.model_version,
            model=self.engine.processor(name).get("model", ""),
            timestamp=None,
            beginTimestamp=self.begin_timestamp,
            endTimestamp=self.end_timestamp,
            hostname=getfqdn(),
        )

        self.nafdoc.add_processor_element(name, lp)

    def add_entities_layer(self):
        """Generate and add all entities in document to entities layer"""
        self.add_layer(ENTITIES_LAYER_TAG)

        current_entity = list()  # Use a list for multiword entities.
        current_entity_orth = list()  # id.

        current_token: int = 1  # Keep track of the token number.
        term_number: int = 1  # Keep track of the term number.
        entity_number: int = 1  # Keep track of the entity number.
        total_tokens: int = 0

        parsing_entity: bool = False

        for _, sentence in enumerate(self.engine.document_sentences(self.doc), start=1):

            entity_gen = self.entities_generator(sentence)
            try:
                next_entity = next(entity_gen)
            except StopIteration:
                next_entity = Entity(start=None, end=None, type=None)

            for token_number, token in enumerate(
                self.engine.sentence_tokens(sentence), start=current_token
            ):
                # Do we need a state change?

                if token_number == next_entity.start:
                    parsing_entity = True

                tid = "t" + str(term_number)
                if parsing_entity:
                    current_entity.append(tid)
                    current_entity_orth.append(
                        normalize_token_orth(self.engine.token_orth(token))
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

                    self.nafdoc.add_entity_element(entity_data, self.language)

                    entity_number += 1
                    current_entity = list()
                    current_entity_orth = list()

                    # Move to the next entity
                    parsing_entity = False
                    try:
                        next_entity = next(entity_gen)
                    except StopIteration:
                        next_entity = Entity(start=None, end=None, type=None)

            if self.engine.token_reset() is False:
                current_token = token_number + 1
                total_tokens = 0
            else:
                current_token = 1
                total_tokens += token_number

        return None

    def add_text_layer(self):
        """Generate and add all words in document to text layer"""
        self.add_layer(TEXT_LAYER_TAG)

        # align the stanza output with the tokenized text
        doc = align_stanza_dict_offsets(self.doc.to_dict(), self.tokenized_text)

        wf_count_prev_sent = 0
        idx_w = 0
        for idx_s, s in enumerate(doc, start=1):
            wf_count_prev_sent += idx_w
            for idx_w, wf in enumerate(s, start=1):
                wf_id = "w" + str(wf_count_prev_sent + idx_w)
                wf_offset = wf["start_char"]
                if self.page_offsets is not None:
                    bins = [0] + [p.endIndex for p in self.page_offsets]
                    page_nr = np.digitize(wf_offset, bins)
                else:
                    page_nr = None
                if self.paragraph_offsets is not None:
                    bins = [0] + [para.endIndex for para in self.paragraph_offsets]
                    paragraph_nr = np.digitize(wf_offset, bins)
                else:
                    paragraph_nr = None
                wf_data = WordformElement(
                    id=wf_id,
                    sent=str(idx_s),
                    para=str(paragraph_nr),
                    page=str(page_nr),
                    offset=str(wf_offset),
                    length=str(len(wf["text"])),
                    xpath=None,
                    text=wf["text"],
                )

                self.nafdoc.add_wf_element(wf_data, self.cdata)

        return None

    def add_terms_layer(self):
        """Generate and add all terms in document to terms layer"""
        self.add_layer(TERMS_LAYER_TAG)

        current_term = list()  # Use a list for multiword expressions.
        current_term_orth = list()  # id.

        current_token: int = 1  # Keep track of the token number.
        term_number: int = 1  # Keep track of the term number.
        total_tokens: int = 0

        for _, sentence in enumerate(self.engine.document_sentences(self.doc), start=1):

            for token_number, token in enumerate(
                self.engine.sentence_tokens(sentence), start=current_token
            ):

                wid = "w" + str(token_number + total_tokens)
                tid = "t" + str(term_number)

                current_term.append(wid)
                current_term_orth.append(normalize_token_orth(self.engine.token_orth(token)))

                # Create TermElement data:
                token_pos = self.engine.token_pos(token)
                # :param bool map_udpos2naf_pos: if True, we use "udpos2nafpos_info"
                # to map the Universal Dependencies pos (https://universaldependencies.org/u/pos/)
                # to the NAF pos tagset
                if self.map_udpos2olia:
                    if token_pos in udpos2nafpos_info.keys():
                        pos_type = udpos2nafpos_info[token_pos]["class"]
                        token_pos = udpos2nafpos_info[token_pos]["olia"]
                    else:
                        logging.info("unknown token pos: " + str(token_pos))
                        pos_type = "open"
                        token_pos = "unknown"
                else:
                    pos_type = "open"

                term_data = TermElement(
                    id=tid,
                    type=pos_type,
                    lemma=remove_illegal_chars(self.engine.token_lemma(token)),
                    pos=token_pos,
                    morphofeat=self.engine.token_tag(token),
                    netype=None,
                    case=None,
                    head=None,
                    component_of=None,
                    compound_type=None,
                    span=current_term,
                    ext_refs=list(),
                    comment=current_term_orth,
                )

                self.nafdoc.add_term_element(
                    term_data, self.layer_to_attributes_to_ignore, self.comments
                )

                # Move to the next term
                term_number += 1
                current_term = list()
                current_term_orth = list()

            # At the end of the sentence,
            # add all the dependencies to the XML structure.
            if self.engine.token_reset() is False:
                current_token = token_number + 1
                total_tokens = 0
            else:
                current_token = 1
                total_tokens += token_number

        return None

    def add_deps_layer(self):
        """Generate and add all dependencies in document to deps layer"""
        self.add_layer(DEPS_LAYER_TAG)

        current_token: int = 1
        total_tokens: int = 0

        for sentence in self.engine.document_sentences(self.doc):

            dependencies_for_sentence = list()

            for token_number, token in enumerate(
                self.engine.sentence_tokens(sentence), start=current_token
            ):
                for dep_data in self.dependencies_to_add(sentence, token, total_tokens):
                    if dep_data not in dependencies_for_sentence:
                        dependencies_for_sentence.append(dep_data)

            for dep_data in dependencies_for_sentence:
                self.nafdoc.add_dependency_element(dep_data, self.comments)

            if self.engine.token_reset() is False:
                current_token = token_number + 1
                total_tokens = 0
            else:
                current_token = 1
                total_tokens += token_number

        return None

    def get_next_mw_id(self):
        """Return multiword id for new multiword"""
        layer = self.nafdoc.layer(MULTIWORDS_LAYER_TAG)
        mw_ids = [int(mw_el.get("id")[2:]) for mw_el in layer.xpath("mw")]
        if mw_ids:
            next_mw_id = max(mw_ids) + 1
        else:
            next_mw_id = 1
        return f"mw{next_mw_id}"

    def create_separable_verb_lemma(self, verb, particle, language):
        """return lemma (particle plus verb) for nl and en"""
        if language == "nl":
            lemma = particle + verb
        elif language == "en":
            lemma = f"{verb}_{particle}"
        else:
            lemma = f"{verb}_{particle}"
        return lemma

    def add_multiwords_layer(self):
        """Generate and add all multiwords in document to multiwords layer"""
        self.add_layer(MULTIWORDS_LAYER_TAG)

        if self.naf_version == "v3":
            logging.info("add_multi_words function only applies to naf version 4")

        supported_languages = {"nl", "en"}
        if self.language not in supported_languages:
            logging.info(
                f"add_multi_words function only implemented for "
                f"{supported_languages}, not for supplied {self.language}"
            )

        tid_to_term = {
            term.get("id"): term for term in self.nafdoc.findall("terms/term")
        }

        for dep in self.nafdoc.deps:

            if dep.get("rfunc") == "compound:prt":

                next_mw_id = self.get_next_mw_id()

                idverb = dep.get("from_term")
                idparticle = dep.get("to_term")

                verb_term_el = tid_to_term[idverb]
                verb = verb_term_el.attrib.get("lemma")
                verb_term_el.set("component_of", next_mw_id)

                particle_term_el = tid_to_term[idparticle]
                particle = particle_term_el.attrib.get("lemma")
                particle_term_el.set("component_of", next_mw_id)

                separable_verb_lemma = self.create_separable_verb_lemma(
                    verb, particle, self.language
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

                self.nafdoc.add_multiword_element(multiword_data)

                # component = etree.SubElement(
                #     mw_element, "component", attrib={"id": c_id}
                # )
                # span = etree.SubElement(component, "span")
                # etree.SubElement(span, "target", attrib={"id": t_id})

        # self.nafdoc.add_multi_words(self.naf_version, self.language)

    def add_raw_layer(self):
        """Generate and add raw text in document to raw layer"""
        self.add_layer(RAW_LAYER_TAG)

        wordforms = self.nafdoc.text

        if len(wordforms) > 0:

            delta = int(wordforms[0]["offset"])
            tokens = [" " * delta + wordforms[0]["text"]]

            for prev_wf, cur_wf in zip(wordforms[:-1], wordforms[1:]):
                prev_start = int(prev_wf["offset"])
                prev_end = prev_start + int(prev_wf["length"])
                cur_start = int(cur_wf["offset"])
                delta = cur_start - prev_end
                # no chars between two token (for example with a dot .)
                if delta == 0:
                    leading_chars = ""
                elif delta >= 1:
                    # 1 or more characters between tokens -> n spaces added
                    leading_chars = " " * delta
                elif delta < 0:
                    logging.warning(
                        f"please check the offsets of {prev_wf['text']} and "
                        f"{cur_wf['text']} (delta of {delta})"
                    )
                tokens.append(leading_chars + cur_wf["text"])

            if self.cdata:
                raw_text = etree.CDATA("".join(tokens))
            else:
                raw_text = "".join(tokens)
        else:
            raw_text = ""

        raw_data = RawElement(text=raw_text)

        self.nafdoc.add_raw_text_element(raw_data)

    def add_chunks_layer(self):
        """Generate and add all chunks in document to chunks layer"""
        self.add_layer(CHUNKS_LAYER_TAG)

        for chunk_data in self.chunk_tuples_for_doc(self.doc):
            self.nafdoc.add_chunk_element(chunk_data, self.comments)
