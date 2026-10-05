"""End-to-end, graph-only model:

1. entities: lexicon (``alias`` edges) of the background KG + GeoNames / NCBI subgraphs, small
   places disambiguated by the place hierarchy (ner.py);
2. relations: Personalized PageRank on the document graph joined with the background KG, plus
   the background KG's prior for the very edge (graph_re.py);
3. n-ary events: maximal cliques of the relation graph restricted to edges whose two nodes share a
   sentence node in the document graph (events.py).

No classifier is trained.  ``fit`` builds the background KG and chooses the walk parameters and
one threshold per relation by grid search, using cross-fitting: the entities and KG evidence for
each training document come from a KG built *without* that document, as at test time.
"""
from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field

from .brat import Document, Entity
from .events import cliques_to_events, gold_events
from .graph_re import WalkParams, ppr_scores, tune_thresholds
from .kg import KnowledgeGraph
from .ner import EntityRecognizer
from .relations import RELATIONS, DocGraph, gold_triples

# Grid explored by ``fit`` (all combinations).
PARAM_GRID = {
    "damping": [0.05, 0.1, 0.3],
    "w_sent": [0.05, 0.3],
    "w_kg": [0.25, 0.5, 1.0],
    "w_geo": [0.0, 1.0],
    "kg_prior": [0.0, 3.0, 10.0],
    "centrality": [0.0, 1.0],
}
EVENT_SCALES = [0.5, 1.0, 1.5, 2.0, 3.0]


def fold_order(docs: list[Document], seed: int = 13) -> list[Document]:
    """Shuffled document order; fold k is ``order[k::folds]``."""
    order = docs[:]
    random.Random(seed).shuffle(order)
    return order


def event_edges(scores, thresholds, scale: float, dg: DocGraph):
    """Relation edges usable as event arguments: accepted at ``scale`` x the relation threshold and
    joined through a common sentence node (a 2-hop path entity - sentence - entity)."""
    return {k for k, s in scores.items()
            if s >= thresholds[k[0]] * scale and set(dg.sent[k[1]]) & set(dg.sent[k[2]])}


@dataclass
class Prediction:
    doc_id: str
    entities: list[Entity]
    node_type: dict[str, str]
    scores: dict[tuple[str, str, str], float]
    triples: set[tuple[str, str, str]] = field(default_factory=set)
    events: list[frozenset] = field(default_factory=list)
    dg: DocGraph | None = None


class GraphKGModel:
    def __init__(self, folds: int = 5, ner_threshold: float = 0.3, gold_entities: bool = False,
                 seed: int = 13, verbose: bool = True) -> None:
        self.folds = folds
        self.ner_threshold = ner_threshold
        self.gold_entities = gold_entities
        self.seed = seed
        self.verbose = verbose
        self.params = WalkParams()
        self.rel_threshold: dict[str, float] = {}
        self.event_scale = 1.0

    def _entities(self, doc: Document, ner: EntityRecognizer) -> list[Entity]:
        return list(doc.entities.values()) if self.gold_entities else ner(doc.text)

    def fit(self, docs: list[Document]) -> "GraphKGModel":
        order = fold_order(docs, self.seed)
        held_out = []  # (doc graph, fold KG, gold triples, gold events)
        for k in range(self.folds):
            held = order[k::self.folds]
            ids = {d.id for d in held}
            kg = KnowledgeGraph.build([d for d in order if d.id not in ids])
            ner = EntityRecognizer(kg, self.ner_threshold)
            for doc in held:
                held_out.append((DocGraph(doc, self._entities(doc, ner)), kg, gold_triples(doc), gold_events(doc)))

        best = None
        keys = list(PARAM_GRID)
        for values in itertools.product(*(PARAM_GRID[k] for k in keys)):
            params = WalkParams(**dict(zip(keys, values)))
            samples = [(ppr_scores(dg, kg, params), gold) for dg, kg, gold, _ in held_out]
            thresholds, f1 = tune_thresholds(samples)
            if best is None or f1 > best[0]:
                best = (f1, params, thresholds, samples)
        f1, self.params, self.rel_threshold, samples = best

        # How permissive the relation graph used for cliques should be.
        best_ev = None
        for scale in EVENT_SCALES:
            tp = n_pred = n_gold = 0
            for (scores, _), (dg, _, _, gold_ev) in zip(samples, held_out):
                edges = event_edges(scores, self.rel_threshold, scale, dg)
                pred = set(cliques_to_events(edges, dg.type))
                gold = {e for e in gold_ev if len(e) > 1}
                tp += len(pred & gold); n_pred += len(pred); n_gold += len(gold)
            ev_f1 = 2 * tp / (n_pred + n_gold) if n_pred + n_gold else 0.0
            if best_ev is None or ev_f1 > best_ev[0]:
                best_ev = (ev_f1, scale)
        self.event_scale = best_ev[1]
        if self.verbose:
            print(f"fit: walk {self.params}, out-of-fold relation F1 {f1:.3f}, "
                  f"event scale {self.event_scale} (out-of-fold event F1 {best_ev[0]:.3f})")

        self.kg = KnowledgeGraph.build(docs)
        self.ner = EntityRecognizer(self.kg, self.ner_threshold)
        return self

    def predict(self, docs: list[Document]) -> list[Prediction]:
        preds = []
        for doc in docs:
            ents = self._entities(doc, self.ner)
            dg = DocGraph(doc, ents)
            preds.append(Prediction(doc.id, ents, dg.type, ppr_scores(dg, self.kg, self.params), dg=dg))
        self.decide(preds)
        return preds

    def decide(self, preds: list[Prediction]) -> None:
        for p in preds:
            p.triples = {k for k, s in p.scores.items() if s >= self.rel_threshold[k[0]]}
            edges = event_edges(p.scores, self.rel_threshold, self.event_scale, p.dg)
            p.events = cliques_to_events(edges, p.node_type)


def to_document(doc: Document, pred: Prediction) -> Document:
    """Ground node-level predictions back onto mentions for BioNLP-ST output."""
    out = Document(doc.id, doc.text, doc.split, doc.lang)
    out.entities = {e.id: e for e in pred.entities}
    dg = pred.dg
    for rel, h, t in sorted(pred.triples):
        mh, mt = min(
            ((a, b) for a in dg.mentions[h] for b in dg.mentions[t]),
            key=lambda ab: abs(ab[0].start - ab[1].start),
        )
        out.relations.append((rel, mh.id, mt.id))
    for ev in pred.events:
        anchor = None
        args = {}
        for role, node in sorted(dict(ev).items()):
            ms = dg.mentions[node]
            m = ms[0] if anchor is None else min(ms, key=lambda x: abs(x.start - anchor))
            anchor = m.start if anchor is None else anchor
            args[role] = m.id
        out.events.append(args)
    return out


__all__ = ["GraphKGModel", "Prediction", "to_document", "fold_order", "RELATIONS"]
