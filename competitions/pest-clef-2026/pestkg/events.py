"""N-ary events as maximal cliques of the binary relation graph.

On the EPOP train+dev annotations 88% of the multi-argument events are cliques of binary relations
(every pair of arguments is linked by a relation) and 83% of gold events are exactly a maximal
clique, so we derive the n-ary layer from the predicted binary edges instead of learning it.
"""
from __future__ import annotations

import networkx as nx

from .brat import ROLE_OF_TYPE, Document
from .ner import entity_node


def cliques_to_events(edges: set[tuple[str, str, str]], node_type: dict[str, str]) -> list[frozenset]:
    g = nx.Graph()
    g.add_edges_from((h, t) for _, h, t in edges)
    events = set()
    for clique in nx.find_cliques(g):
        if len(clique) < 2:
            continue
        roles = [ROLE_OF_TYPE[node_type[n]] for n in clique]
        if len(set(roles)) != len(roles):  # one argument per role
            continue
        events.add(frozenset(zip(roles, clique)))
    return sorted(events, key=sorted)


def gold_events(doc: Document) -> set[frozenset]:
    out = set()
    for ev in doc.events:
        if all(e in doc.entities for e in ev.values()):
            out.add(frozenset((r, entity_node(doc.entities[e])) for r, e in ev.items()))
    return out


ROLES = sorted(set(ROLE_OF_TYPE.values()))
EVENT_FEATURES = (
    ["size"] + [f"role:{r}" for r in ROLES]
    + ["min_edge", "mean_edge", "max_edge", "maximal_high", "maximal_low", "extensions",
       "shared_sents", "max_sent_span", "top_ranked", "min_mentions"]
)


def event_candidates(scores: dict, node_type: dict, dg, threshold: dict, low: float = 0.5, max_size: int = 5):
    """Cliques of the relation graph as event candidates, with features for a scorer.

    Edge weight = relation score / its decision threshold, so 1.0 means "just accepted".  The low
    graph keeps edges down to ``low``; the high graph is the accepted relations.
    """
    w: dict[frozenset, float] = {}
    for (rel, h, t), s in scores.items():
        r = s / threshold[rel]
        if r >= low:
            key = frozenset((h, t))
            w[key] = max(w.get(key, 0.0), r)
    g_low = nx.Graph()
    g_low.add_edges_from(tuple(k) for k in w)
    g_high = nx.Graph()
    g_high.add_edges_from(tuple(k) for k, r in w.items() if r >= 1.0)
    max_low = {frozenset(c) for c in nx.find_cliques(g_low)}
    max_high = {frozenset(c) for c in nx.find_cliques(g_high)} if g_high.number_of_edges() else set()
    out = []
    for clique in nx.enumerate_all_cliques(g_low):
        if len(clique) < 2:
            continue
        if len(clique) > max_size:
            break
        roles = [ROLE_OF_TYPE[node_type[n]] for n in clique]
        if len(set(roles)) != len(roles):
            continue
        cs = frozenset(clique)
        ew = [w[frozenset((a, b))] for i, a in enumerate(clique) for b in clique[i + 1:]]
        common = None
        for n in clique:
            nb = set(g_high.neighbors(n)) if n in g_high else set()
            common = nb if common is None else common & nb
        sents = [set(dg.sent[n]) for n in clique]
        shared = set.intersection(*sents)
        firsts = [min(s) for s in sents]
        feats = (
            [len(clique)] + [int(r in roles) for r in ROLES]
            + [min(ew), sum(ew) / len(ew), max(ew), int(cs in max_high), int(cs in max_low),
               len(common or ()), len(shared), max(firsts) - min(firsts),
               sum(dg.rank[n] == 0 for n in clique) / len(clique),
               min(len(dg.mentions[n]) for n in clique)]
        )
        out.append((frozenset(zip(roles, clique)), feats))
    return out
