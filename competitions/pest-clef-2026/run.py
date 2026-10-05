"""PestCLEF 2026 — graph-only knowledge-graph pipeline.

    python run.py dev                 # train on train, evaluate on dev
    python run.py dev --gold-entities # relation/event extraction on gold entities (RE upper bound)
    python run.py test                # train on train+dev, predict test, write submission files
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from pestkg.brat import load_split, write_a2
from pestkg.evaluate import evaluate, format_report
from pestkg.pipeline import GraphKGModel, to_document

HERE = Path(__file__).parent
DATA = HERE / "data"


def load(split: str):
    return load_split(DATA / "EPOP_documents", DATA / "EPOP_annotations", split)


def write_submission(docs, preds, model, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "a2").mkdir(exist_ok=True)
    kg = model.kg
    with (out / "relations.csv").open("w", newline="", encoding="utf8") as fh:
        w = csv.writer(fh)
        w.writerow(["doc_id", "relation", "head_type", "head_id", "head_text",
                    "tail_type", "tail_id", "tail_text", "score"])
        for doc, p in zip(docs, preds):
            for rel, h, t in sorted(p.triples):
                w.writerow([doc.id, rel, p.node_type[h], h, p.dg.mentions[h][0].text,
                            p.node_type[t], t, p.dg.mentions[t][0].text, round(p.scores[(rel, h, t)], 4)])
    with (out / "events.csv").open("w", newline="", encoding="utf8") as fh:
        w = csv.writer(fh)
        w.writerow(["doc_id", "event", "arguments"])
        for doc, p in zip(docs, preds):
            for i, ev in enumerate(p.events, 1):
                args = {role: {"id": n, "text": p.dg.mentions[n][0].text} for role, n in sorted(ev)}
                w.writerow([doc.id, f"E{i}", json.dumps(args, ensure_ascii=False)])
    for doc, p in zip(docs, preds):
        write_a2(to_document(doc, p), out / "a2" / f"{doc.id}.a2")
    kg.save(out / "kg")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["dev", "test"])
    ap.add_argument("--gold-entities", action="store_true")
    ap.add_argument("--ner-threshold", type=float, default=0.3)
    ap.add_argument("--out", type=Path, default=HERE / "outputs")
    args = ap.parse_args()

    train, dev = load("train"), load("dev")
    if args.mode == "dev":
        model = GraphKGModel(ner_threshold=args.ner_threshold, gold_entities=args.gold_entities).fit(train)
        preds = model.predict(dev)
        print("== dev (parameters and thresholds chosen out-of-fold on train)")
        print(format_report(evaluate(dev, preds)))
        write_submission(dev, preds, model, args.out / "dev")
    else:
        model = GraphKGModel(ner_threshold=args.ner_threshold).fit(train + dev)
        test = load("test")
        preds = model.predict(test)
        write_submission(test, preds, model, args.out / "test")
        n_rel = sum(len(p.triples) for p in preds)
        n_ev = sum(len(p.events) for p in preds)
        print(f"test: {len(test)} docs, {n_rel} relations, {n_ev} events -> {args.out / 'test'}")


if __name__ == "__main__":
    main()
