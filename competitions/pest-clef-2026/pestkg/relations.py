"""Relation schema and the per-document graph of entity nodes.

Mentions of the same taxon / place / disease are merged into one node; every pair of nodes whose
types fit a relation signature of the schema (the ontology of the task) is a candidate edge.
Candidates are scored by graph_re.py.
"""
from __future__ import annotations

import bisect
from collections import defaultdict

from .brat import Document, Entity
from .ner import entity_node

SIGNATURES = {
    "Located_in": ({"Pest", "Plant", "Disease", "Vector", "Dissemination_pathway"}, {"Location"}),
    "Detected_on": ({"Pest", "Plant", "Disease", "Vector", "Dissemination_pathway", "Location"}, {"Date"}),
    "Found_on": ({"Pest", "Vector"}, {"Plant", "Dissemination_pathway"}),
    "Causes": ({"Pest"}, {"Disease"}),
    "Expressed_by": ({"Disease"}, {"Plant"}),
    "Dispersed_by": ({"Pest", "Vector", "Disease"}, {"Dissemination_pathway"}),
    "Vected_by": ({"Pest", "Disease"}, {"Vector"}),
}
RELATIONS = list(SIGNATURES)


class DocGraph:
    """Nodes of one document with their mentions and sentence positions."""

    def __init__(self, doc: Document, entities: list[Entity]) -> None:
        self.doc = doc
        self.bounds = _sentence_starts(doc.text)
        self.mentions: dict[str, list[Entity]] = defaultdict(list)
        self.type: dict[str, str] = {}
        for e in entities:
            n = entity_node(e)
            self.mentions[n].append(e)
            self.type.setdefault(n, e.type)
        self.sent = {n: [self.sentence(m.start) for m in ms] for n, ms in self.mentions.items()}

    def sentence(self, offset: int) -> int:
        return bisect.bisect_right(self.bounds, offset) - 1

    def candidates(self):
        for rel, (heads, tails) in SIGNATURES.items():
            for h, ht in self.type.items():
                if ht not in heads:
                    continue
                for t, tt in self.type.items():
                    if tt in tails and t != h:
                        yield rel, h, t


def _sentence_starts(text: str) -> list[int]:
    from .text import sentence_bounds

    return sentence_bounds(text)


def gold_triples(doc: Document) -> set[tuple[str, str, str]]:
    out = set()
    for rel, h, t in doc.relations:
        if h in doc.entities and t in doc.entities:
            out.add((rel, entity_node(doc.entities[h]), entity_node(doc.entities[t])))
    return out
