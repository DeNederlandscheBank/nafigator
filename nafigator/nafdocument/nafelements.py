from dataclasses import dataclass
from typing import List, Tuple
from datetime import datetime


@dataclass
class ProcessorElement:
    name: str
    version: str
    model: str
    timestamp: datetime
    beginTimestamp: datetime
    endTimestamp: datetime
    hostname: str


@dataclass
class WordformElement:
    id: str
    sent: str
    para: str
    page: str
    offset: str
    length: str
    xpath: str
    text: str


@dataclass
class TermElement:
    id: str
    type: str
    lemma: str
    pos: str
    morphofeat: str
    netype: str
    case: str
    head: str
    component_of: str
    compound_type: str
    span: List[str]
    ext_refs: List[dict]
    comment: List[str]


@dataclass
class Entity:
    start: str
    end: str
    type: str


@dataclass
class EntityElement:
    id: str
    type: str
    status: str
    source: str
    span: List[str]
    ext_refs: List[dict]
    comment: List[str]


@dataclass
class DependencyRelation:
    from_term: str
    to_term: str
    rfunc: str
    case: str
    comment: str


@dataclass
class ChunkElement:
    id: str
    head: str
    phrase: str
    case: str
    span: List[str]
    comment: str


@dataclass
class RawElement:
    text: str


@dataclass
class MultiwordElement:
    id: str
    lemma: str
    pos: str
    morphofeat: str
    case: str
    status: str
    type: str
    components: List[Tuple[str, str]]


@dataclass
class ComponentElement:
    id: str
    type: str
    lemma: str
    pos: str
    morphofeat: str
    netype: str
    case: str
    head: str
    span: List[str]
