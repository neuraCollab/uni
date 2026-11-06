"""PestCLEF 2026 — graph-only knowledge-graph pipeline.

    python run.py dev                 # train on train, evaluate on dev
    python run.py dev --gold-entities # relation/event extraction on gold entities (RE upper bound)
    python run.py test                # train on train+dev, predict test, write submission files

The Kaggle submission is outputs/test/submission.csv. If the competition's JSON files are in
data/kaggle/ (or --kaggle-dir), dev is also scored with the Kaggle metric and the test document
ids are taken from PestCLEF-2026_test.json.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from pestkg import kaggle
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
    ap.add_argument("--kaggle-dir", type=Path, default=DATA / "kaggle",
                    help="folder with the competition's PestCLEF-2026_{dev,test}.json (optional)")
    args = ap.parse_args()

    train, dev = load("train"), load("dev")
    if args.mode == "dev":
        model = GraphKGModel(ner_threshold=args.ner_threshold, gold_entities=args.gold_entities).fit(train)
        preds = model.predict(dev)
        print("== dev (parameters and thresholds chosen out-of-fold on train)")
        print(format_report(evaluate(dev, preds)))
        edges = {p.doc_id: kaggle.kaggle_edges(p) for p in preds}
        dev_json = args.kaggle_dir / "PestCLEF-2026_dev.json"
        if dev_json.exists():
            res = kaggle.score(kaggle.load_gold(dev_json), edges)
            print(f"== dev, Kaggle metric: P {res['P']:.3f}  R {res['R']:.3f}  F1 {res['F1']:.3f}"
                  f"  ({res['tp']} correct of {res['pred']} predicted, {res['gold']} gold edges)")
        write_submission(dev, preds, model, args.out / "dev")
        kaggle.write_submission(args.out / "dev" / "submission.csv", [d.id for d in dev], edges)
    else:
        model = GraphKGModel(ner_threshold=args.ner_threshold).fit(train + dev)
        test = load("test")
        preds = model.predict(test)
        write_submission(test, preds, model, args.out / "test")
        edges = {p.doc_id: kaggle.kaggle_edges(p) for p in preds}
        test_json = args.kaggle_dir / "PestCLEF-2026_test.json"
        doc_ids = kaggle.load_doc_ids(test_json) if test_json.exists() else [d.id for d in test]
        missing = set(doc_ids) - {d.id for d in test}
        if missing:
            print(f"warning: no text for {len(missing)} test documents, they get an empty graph")
        kaggle.write_submission(args.out / "test" / "submission.csv", doc_ids, edges)
        n_rel = sum(len(e) for e in edges.values())
        n_ev = sum(len(p.events) for p in preds)
        print(f"test: {len(doc_ids)} docs, {n_rel} edges, {n_ev} events -> {args.out / 'test' / 'submission.csv'}")


if __name__ == "__main__":
    main()
