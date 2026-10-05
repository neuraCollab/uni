"""Document-level relation extraction between knowledge-graph nodes.

A document is turned into a small graph: its nodes are the entities found in it (mentions of the
same taxon / place / disease are merged into one node) and every pair of nodes whose types fit a
relation signature is a candidate edge.  Each candidate is described by textual evidence
(co-occurrence in sentences, distance, cue words, salience of the nodes in the document) and by
evidence from the background knowledge graph (known edge, known edges of the head, 2-hop paths,
place hierarchy).  A gradient-boosted classifier scores the candidates.
"""
from __future__ import annotations

import bisect
import math
import re
from collections import Counter, defaultdict

import numpy as np

from .brat import Document, Entity
from .kg import KnowledgeGraph
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
TYPES = ["Pest", "Plant", "Location", "Date", "Disease", "Vector", "Dissemination_pathway"]
CUES = {
    "Located_in": r"\b(in|at|from|of|across|throughout|present|found|detected|reported|outbreak)\b",
    "Detected_on": r"\b(detected|found|reported|since|first|in|identified|confirmed)\b",
    "Found_on": r"\b(on|in|infect\w*|attack\w*|host\w*|affect\w*|damag\w*|feed\w*|of)\b",
    "Causes": r"\b(caus\w*|agent|responsible|produc\w*|known as|called)\b",
    "Expressed_by": r"\b(on|in|of|affect\w*|symptom\w*|infect\w*)\b",
    "Dispersed_by": r"\b(through|via|by|with|import\w*|spread\w*|transport\w*|move\w*|carr\w*)\b",
    "Vected_by": r"\b(vector\w*|transmit\w*|spread\w*|carr\w*|by|insect\w*)\b",
}
CUE_RE = {r: re.compile(p, re.I) for r, p in CUES.items()}


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
        self.n_sent = len(self.bounds)
        freq = Counter({n: len(ms) for n, ms in self.mentions.items()})
        self.rank = {}
        for t in TYPES:
            ordered = sorted((n for n in freq if self.type[n] == t), key=lambda n: -freq[n])
            for r, n in enumerate(ordered):
                self.rank[n] = r
        self.n_type = Counter(self.type.values())
        title_end = doc.text.find("\n") if "\n" in doc.text[:400] else 200
        self.in_title = {n: int(any(m.start < title_end for m in ms)) for n, ms in self.mentions.items()}
        self.first = {n: min(m.start for m in ms) / max(len(doc.text), 1) for n, ms in self.mentions.items()}

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


def featurize(dg: DocGraph, kg: KnowledgeGraph, rel: str, h: str, t: str) -> list[float]:
    text = dg.doc.text
    hs, ts = dg.sent[h], dg.sent[t]
    hset, tset = set(hs), set(ts)
    shared = hset & tset
    sdist = min(abs(a - b) for a in hset for b in tset)
    # Closest mention pair and the text between them.
    best = None
    for mh in dg.mentions[h]:
        for mt in dg.mentions[t]:
            d = mt.start - mh.end if mt.start >= mh.end else mh.start - mt.end
            if best is None or d < best[0]:
                best = (d, mh, mt)
    d, mh, mt = best
    between = text[min(mh.end, mt.end):max(mh.start, mt.start)]
    cue = int(bool(CUE_RE[rel].search(between))) if len(between) < 200 else 0
    tail_before = int(mt.start < mh.start)
    # How often is t the nearest node of its type to a mention of h (and vice versa)?
    nearest_t = _nearest_share(dg, h, t)
    nearest_h = _nearest_share(dg, t, h)
    # Background-graph evidence.
    e_count = kg.edges.get((rel, h, t), 0)
    feats = [
        RELATIONS.index(rel), TYPES.index(dg.type[h]), TYPES.index(dg.type[t]),
        len(shared), len(shared) / max(len(hset | tset), 1), sdist, math.log1p(max(d, 0)),
        cue, tail_before, int(bool(re.search(r"[;:]|\n", between))) if d < 300 else 1,
        nearest_t, nearest_h,
        len(dg.mentions[h]), len(dg.mentions[t]), dg.rank[h], dg.rank[t],
        dg.n_type[dg.type[h]], dg.n_type[dg.type[t]], dg.in_title[h], dg.in_title[t],
        dg.first[h], dg.first[t], dg.n_sent,
        math.log1p(e_count), int(e_count > 0),
        math.log1p(kg.out_rel.get((rel, h), 0)), math.log1p(kg.in_rel.get((rel, t), 0)),
        math.log1p(kg.two_hop(h, t)), math.log1p(kg.degree(h)), math.log1p(kg.degree(t)),
        kg.geo_related(rel, h, t) if dg.type[t] == "Location" else -1,
        int(any(kg.edges.get((r, h, t), 0) or kg.edges.get((r, t, h), 0) for r in RELATIONS if r != rel)),
    ]
    return feats


FEATURE_NAMES = [
    "rel", "head_type", "tail_type", "shared_sents", "shared_ratio", "sent_dist", "log_char_dist",
    "cue_between", "tail_before", "punct_between", "t_nearest_to_h", "h_nearest_to_t",
    "h_mentions", "t_mentions", "h_rank", "t_rank", "n_head_type", "n_tail_type", "h_title", "t_title",
    "h_first", "t_first", "n_sent", "kg_log_edge", "kg_edge", "kg_head_out", "kg_tail_in",
    "kg_two_hop", "kg_deg_h", "kg_deg_t", "kg_geo", "kg_other_rel",
]


def _nearest_share(dg: DocGraph, a: str, b: str) -> float:
    """Share of mentions of ``a`` for which ``b`` is the closest node of ``b``'s type."""
    btype = dg.type[b]
    others = [(m.start, n) for n, ms in dg.mentions.items() if dg.type[n] == btype for m in ms]
    if not others:
        return 0.0
    hits = 0
    for ma in dg.mentions[a]:
        nearest = min(others, key=lambda x: abs(x[0] - ma.start))[1]
        hits += nearest == b
    return hits / len(dg.mentions[a])


def gold_triples(doc: Document) -> set[tuple[str, str, str]]:
    out = set()
    for rel, h, t in doc.relations:
        if h in doc.entities and t in doc.entities:
            out.add((rel, entity_node(doc.entities[h]), entity_node(doc.entities[t])))
    return out


def build_matrix(doc: Document, entities: list[Entity], kg: KnowledgeGraph, with_labels: bool = True):
    dg = DocGraph(doc, entities)
    gold = gold_triples(doc) if with_labels else set()
    keys, X, y = [], [], []
    for rel, h, t in dg.candidates():
        keys.append((rel, h, t))
        X.append(featurize(dg, kg, rel, h, t))
        y.append(int((rel, h, t) in gold))
    return dg, keys, np.array(X, dtype=float).reshape(len(X), len(FEATURE_NAMES)), np.array(y)
