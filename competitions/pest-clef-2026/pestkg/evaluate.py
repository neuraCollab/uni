"""Micro P/R/F1 at the knowledge-graph level (text grounding is not required by the task)."""
from __future__ import annotations

from collections import Counter

from .brat import Document
from .events import gold_events
from .ner import entity_node
from .relations import gold_triples


def prf(tp: int, n_pred: int, n_gold: int) -> dict[str, float]:
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"P": round(p, 4), "R": round(r, 4), "F1": round(f, 4), "tp": tp, "pred": n_pred, "gold": n_gold}


def evaluate(docs: list[Document], preds) -> dict[str, dict]:
    c = Counter()
    per_rel = Counter()
    for doc, p in zip(docs, preds):
        g_spans = {(e.start, e.end, e.type) for e in doc.entities.values()}
        p_spans = {(e.start, e.end, e.type) for e in p.entities}
        c["men_tp"] += len(g_spans & p_spans); c["men_p"] += len(p_spans); c["men_g"] += len(g_spans)
        g_nodes = {(e.type, entity_node(e)) for e in doc.entities.values()}
        p_nodes = {(e.type, entity_node(e)) for e in p.entities}
        c["node_tp"] += len(g_nodes & p_nodes); c["node_p"] += len(p_nodes); c["node_g"] += len(g_nodes)
        g_tr, p_tr = gold_triples(doc), p.triples
        c["rel_tp"] += len(g_tr & p_tr); c["rel_p"] += len(p_tr); c["rel_g"] += len(g_tr)
        for rel, *_ in g_tr & p_tr:
            per_rel[(rel, "tp")] += 1
        for rel, *_ in p_tr:
            per_rel[(rel, "p")] += 1
        for rel, *_ in g_tr:
            per_rel[(rel, "g")] += 1
        g_ev, p_ev = gold_events(doc), set(p.events)
        c["ev_tp"] += len(g_ev & p_ev); c["ev_p"] += len(p_ev); c["ev_g"] += len(g_ev)
    out = {
        "mentions": prf(c["men_tp"], c["men_p"], c["men_g"]),
        "nodes": prf(c["node_tp"], c["node_p"], c["node_g"]),
        "relations": prf(c["rel_tp"], c["rel_p"], c["rel_g"]),
        "events": prf(c["ev_tp"], c["ev_p"], c["ev_g"]),
    }
    rels = sorted({r for r, _ in per_rel})
    for r in rels:
        out[f"rel:{r}"] = prf(per_rel[(r, "tp")], per_rel[(r, "p")], per_rel[(r, "g")])
    return out


def format_report(res: dict[str, dict]) -> str:
    lines = [f"{'level':<22}{'P':>8}{'R':>8}{'F1':>8}{'pred':>7}{'gold':>7}"]
    for k, v in res.items():
        lines.append(f"{k:<22}{v['P']:>8.3f}{v['R']:>8.3f}{v['F1']:>8.3f}{v['pred']:>7}{v['gold']:>7}")
    return "\n".join(lines)
