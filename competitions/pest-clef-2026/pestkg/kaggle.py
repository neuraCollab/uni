"""Kaggle PestCLEF 2026 format: predicate names, gold graph, scorer and submission CSV.

The Kaggle release renames three EPOP relations (``Detected_on`` -> ``Occurs_on``,
``Expressed_by`` -> ``Affects``, ``Vected_by(pest, vector)`` -> ``Transmits(vector, pest)``, i.e.
reversed).  A gold edge lists every accepted surface form of its subject and object node, where a
node is a group of mentions sharing the same set of ontology references (or a single unnormalised
mention), merged further by identity coreference.  A predicted edge is a single string per node;
it matches a gold edge with the same predicate when both strings are accepted forms.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

PREDICATE = {
    "Located_in": "Located_in",
    "Detected_on": "Occurs_on",
    "Found_on": "Found_on",
    "Causes": "Causes",
    "Expressed_by": "Affects",
    "Dispersed_by": "Dispersed_by",
    "Vected_by": "Transmits",
}
REVERSED = {"Vected_by"}  # Kaggle's Transmits goes vector -> pest

GoldEdge = tuple[str, frozenset, frozenset]


def load_gold(path: Path) -> dict[str, list[GoldEdge]]:
    data = json.loads(Path(path).read_text(encoding="utf8"))
    return {
        doc["doc_id"]: [(e["predicate"], frozenset(e["subject"]), frozenset(e["object"]))
                        for e in doc.get("knowledge_graph", [])]
        for doc in data
    }


def load_doc_ids(path: Path) -> list[str]:
    return [doc["doc_id"] for doc in json.loads(Path(path).read_text(encoding="utf8"))]


def node_form(dg, node: str) -> str:
    """The surface form printed for a node: its most frequent mention, earliest on ties."""
    counts = Counter(m.text for m in dg.mentions[node])
    first = {}
    for m in sorted(dg.mentions[node], key=lambda m: m.start):
        first.setdefault(m.text, m.start)
    return min(counts, key=lambda t: (-counts[t], first[t]))


def kaggle_edges(pred) -> list[tuple[str, str, str]]:
    """Predicted triples of one document as Kaggle (predicate, subject, object) strings."""
    out = []
    for rel, h, t in sorted(pred.triples, key=lambda k: -pred.scores[k]):
        s, o = node_form(pred.dg, h), node_form(pred.dg, t)
        if rel in REVERSED:
            s, o = o, s
        edge = (PREDICATE[rel], s, o)
        if edge not in out:
            out.append(edge)
    return out


def score(gold: dict[str, list[GoldEdge]], predicted: dict[str, list[tuple[str, str, str]]],
          ignore_case: bool = False) -> dict:
    """Micro P/R/F1; each gold edge can be matched by one predicted edge."""
    norm = (lambda s: s.lower()) if ignore_case else (lambda s: s)
    tp = n_pred = n_gold = 0
    per_pred = Counter()
    for doc_id, edges in gold.items():
        g = [(p, {norm(x) for x in s}, {norm(x) for x in o}) for p, s, o in edges]
        used = [False] * len(g)
        n_gold += len(g)
        for p, s, o in predicted.get(doc_id, []):
            n_pred += 1
            per_pred[(p, "pred")] += 1
            for i, (gp, gs, go) in enumerate(g):
                if not used[i] and gp == p and norm(s) in gs and norm(o) in go:
                    used[i] = True
                    tp += 1
                    per_pred[(p, "tp")] += 1
                    break
        for p, _, _ in edges:
            per_pred[(p, "gold")] += 1
    prec = tp / n_pred if n_pred else 0.0
    rec = tp / n_gold if n_gold else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    out = {"P": round(prec, 4), "R": round(rec, 4), "F1": round(f1, 4), "tp": tp, "pred": n_pred, "gold": n_gold}
    out["by_predicate"] = {
        p: {"tp": per_pred[(p, "tp")], "pred": per_pred[(p, "pred")], "gold": per_pred[(p, "gold")]}
        for p in sorted({k[0] for k in per_pred})
    }
    return out


def write_submission(path: Path, doc_ids: list[str], predicted: dict[str, list[tuple[str, str, str]]]) -> None:
    """``doc_id,knowledge_graph`` CSV; every listed document gets a row (``[]`` when empty)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf8") as fh:
        w = csv.writer(fh, quoting=csv.QUOTE_ALL)
        w.writerow(["doc_id", "knowledge_graph"])
        for doc_id in doc_ids:
            edges = [{"predicate": p, "subject": s, "object": o} for p, s, o in predicted.get(doc_id, [])]
            w.writerow([doc_id, json.dumps(edges, ensure_ascii=False)])
