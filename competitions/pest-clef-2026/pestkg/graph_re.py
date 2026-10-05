"""Relation extraction by random walks on a document graph joined with the background KG.

For one document we build a weighted undirected graph whose nodes are

* the document's entities (KG nodes: a taxon, a place, a date ...),
* its sentences ``("S", i)``,

and whose edges are

* ``entity — sentence``  (the entity is mentioned in the sentence; weight = number of mentions),
* ``sentence — next sentence``  (reading order; weight ``w_sent``),
* ``entity — entity``  when the background KG already links the two nodes by any relation
  (weight ``w_kg * log(1 + #documents asserting it)``),
* ``place — place``  when GeoNames puts one inside the other (weight ``w_geo``).

The score of a candidate edge ``(rel, h, t)`` is the Personalized PageRank of ``t`` for a walk
restarting at ``h`` (all PPR vectors of a document come from one matrix inverse), divided by the
best score among the nodes of ``t``'s type, so that it reads "how close is ``t`` to ``h``
compared with its competitors".  A relation is accepted when this score passes a per-relation
threshold.  The ontology (allowed head/tail types of each relation) restricts the candidates.
There is no learned classifier: the handful of graph parameters and thresholds are chosen by
grid search on training documents.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .kg import KnowledgeGraph
from .relations import RELATIONS, DocGraph


@dataclass(frozen=True)
class WalkParams:
    damping: float = 0.7  # probability of following an edge (1 - restart probability)
    w_sent: float = 0.3
    w_kg: float = 1.0
    w_geo: float = 1.0
    kg_prior: float = 0.0   # boost x (1 + kg_prior * log(1 + #docs with this very edge in the background KG))
    centrality: float = 0.0  # boost x (PageRank of the head / best PageRank of its type) ** centrality
    norm: str = "sym"  # "max": divide by best node of the tail type; "raw": PPR x #nodes; "sym": sqrt(h->t * t->h) x #nodes


def doc_matrix(dg: DocGraph, kg: KnowledgeGraph, p: WalkParams):
    ents = list(dg.type)
    sents = sorted({s for ss in dg.sent.values() for s in ss})
    index = {n: i for i, n in enumerate(ents)}
    for s in sents:
        index[("S", s)] = len(index)
    n = len(index)
    W = np.zeros((n, n))

    def add(a, b, w):
        if w > 0:
            i, j = index[a], index[b]
            W[i, j] += w
            W[j, i] += w

    for e, ss in dg.sent.items():
        for s in ss:
            add(e, ("S", s), 1.0)
    for a, b in zip(sents, sents[1:]):
        add(("S", a), ("S", b), p.w_sent / max(b - a, 1))
    if p.w_kg or p.w_geo:
        for i, a in enumerate(ents):
            anc_a = set(kg.ancestors(a)) if dg.type[a] == "Location" else ()
            for b in ents[i + 1:]:
                c = kg.adj.get(a, {}).get(b, 0)
                if c:
                    add(a, b, p.w_kg * math.log1p(c))
                if anc_a and dg.type[b] == "Location" and (b in anc_a or a in kg.ancestors(b)):
                    add(a, b, p.w_geo)
    return ents, index, W


def ppr_scores(dg: DocGraph, kg: KnowledgeGraph, p: WalkParams) -> dict[tuple[str, str, str], float]:
    """Max-normalised PPR score for every ontology-compatible candidate edge of the document."""
    if not dg.type:
        return {}
    ents, index, W = doc_matrix(dg, kg, p)
    deg = W.sum(1)
    deg[deg == 0] = 1.0
    P = W / deg[:, None]  # row-stochastic transition matrix
    n = len(W)
    # Column j of R is the PPR vector of a walk restarting at node j.
    R = (1 - p.damping) * np.linalg.inv(np.eye(n) - p.damping * P.T)
    by_type: dict[str, list[str]] = {}
    for e in ents:
        by_type.setdefault(dg.type[e], []).append(e)
    cent = {}
    if p.centrality:
        pr = R.mean(axis=1)  # PageRank with uniform restart = average of all PPR vectors
        for t, members in by_type.items():
            top = max(pr[index[m]] for m in members)
            for m in members:
                cent[m] = (pr[index[m]] / top) ** p.centrality if top > 0 else 1.0
    out = {}
    for rel, h, t in dg.candidates():
        col = R[:, index[h]]
        if p.norm == "max":
            best = max(col[index[x]] for x in by_type[dg.type[t]] if x != h)
            out[(rel, h, t)] = float(col[index[t]] / best) if best > 0 else 0.0
        elif p.norm == "raw":
            out[(rel, h, t)] = float(col[index[t]] * n)
        else:
            out[(rel, h, t)] = float(math.sqrt(max(col[index[t]] * R[index[h], index[t]], 0)) * n)
        if p.kg_prior:
            out[(rel, h, t)] *= 1 + p.kg_prior * math.log1p(kg.edges.get((rel, h, t), 0))
        if p.centrality:
            out[(rel, h, t)] *= cent[h]
    return out


def tune_thresholds(samples, grid=None) -> tuple[dict[str, float], float]:
    """samples: list of (scores dict, gold triple set).  Returns per-relation thresholds and the
    micro F1 they reach; gold triples whose nodes the NER missed count as false negatives."""
    if grid is None:
        vals = sorted(s for scores, _ in samples for s in scores.values())
        grid = sorted({vals[int(q * (len(vals) - 1))] for q in np.linspace(0.3, 0.999, 120)}) if vals else [1.0]
    thresholds, tp_all, pred_all, gold_all = {}, 0, 0, 0
    for rel in RELATIONS:
        pairs = [(s, k in gold) for scores, gold in samples for k, s in scores.items() if k[0] == rel]
        n_gold = sum(1 for _, gold in samples for k in gold if k[0] == rel)
        best = (-1.0, 1.0, 0, 0)
        for th in grid:
            pred = [y for s, y in pairs if s >= th]
            tp = sum(pred)
            f1 = 2 * tp / (len(pred) + n_gold) if len(pred) + n_gold else 0.0
            if f1 > best[0]:
                best = (f1, th, tp, len(pred))
        thresholds[rel] = best[1]
        tp_all += best[2]
        pred_all += best[3]
        gold_all += n_gold
    f1 = 2 * tp_all / (pred_all + gold_all) if pred_all + gold_all else 0.0
    return thresholds, f1
