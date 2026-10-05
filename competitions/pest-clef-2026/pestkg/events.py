"""N-ary events as maximal cliques of the binary relation graph.

On the EPOP train+dev annotations 88% of the multi-argument events are cliques of binary relations
(every pair of arguments is linked by a relation) and 83% of gold events are exactly a maximal
clique, so the n-ary layer is read off the graph of predicted binary relations.
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
