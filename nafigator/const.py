# coding: utf-8

"""Const module.

This module contains constants for nafigator package

"""

from collections import namedtuple

ProcessorElement = namedtuple(
    "lp", "name version model timestamp beginTimestamp endTimestamp hostname"
)

WordformElement = namedtuple("WfElement", "id sent para page offset length xpath text")

TermElement = namedtuple(
    "TermElement",
    "id type lemma pos morphofeat netype case head component_of compound_type span ext_refs comment",
)

Entity = namedtuple("Entity", "start end type")

EntityElement = namedtuple(
    "EntityElement", "id type status source span ext_refs comment"
)

DependencyRelation = namedtuple(
    "DependencyRelation", "from_term to_term rfunc case comment"
)

ChunkElement = namedtuple("ChunkElement", "id head phrase case span comment")

RawElement = namedtuple("RawElement", "text")

MultiwordElement = namedtuple(
    "MultiwordElement", "id lemma pos morphofeat case status type components"
)

ComponentElement = namedtuple(
    "ComponentElement", "id type lemma pos morphofeat netype case head span"
)

hidden_characters = ["\a", "\b", "\t", "\n", "\v", "\f", "\r"]

hidden_table = {ord(hidden_character): " " for hidden_character in hidden_characters}
