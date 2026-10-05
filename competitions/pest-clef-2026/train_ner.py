"""Fine-tune the transformer NER (GPU if available, CPU otherwise).

    python train_ner.py --model dmis-lab/biobert-base-cased-v1.2 --train train --eval dev
    python train_ner.py --model dmis-lab/biobert-base-cased-v1.2 --train train,dev   # for the test run

The model is saved to data/models/<model-name>__<train splits>/ and picked up by
``run.py --ner-model``.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from pestkg.evaluate import format_report, prf
from pestkg.kg import KnowledgeGraph
from pestkg.ner import EntityRecognizer, entity_node
from pestkg.neural_ner import NeuralNER, NeuralRecognizer
from run import DATA, load


def entity_report(docs, recognizer) -> dict:
    c = dict(mt=0, mp=0, mg=0, nt=0, np=0, ng=0)
    for d in docs:
        pred = recognizer(d.text)
        gs = {(e.start, e.end, e.type) for e in d.entities.values()}
        ps = {(e.start, e.end, e.type) for e in pred}
        gn = {(e.type, entity_node(e)) for e in d.entities.values()}
        pn = {(e.type, entity_node(e)) for e in pred}
        c["mt"] += len(gs & ps); c["mp"] += len(ps); c["mg"] += len(gs)
        c["nt"] += len(gn & pn); c["np"] += len(pn); c["ng"] += len(gn)
    return {"mentions": prf(c["mt"], c["mp"], c["mg"]), "nodes": prf(c["nt"], c["np"], c["ng"])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dmis-lab/biobert-base-cased-v1.2")
    ap.add_argument("--train", default="train")
    ap.add_argument("--eval", default="")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    splits = args.train.split(",")
    train = [d for s in splits for d in load(s)]
    out = args.out or DATA / "models" / f"{args.model.split('/')[-1]}__{'+'.join(splits)}"
    t0 = time.time()
    ner = NeuralNER(args.model, max_length=args.max_length)
    print(f"device: {ner.device}, {len(train)} training docs")
    ner.fit(train, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
    ner.save(out)
    print(f"saved {out} ({time.time() - t0:.0f}s)")

    if args.eval:
        docs = load(args.eval)
        kg = KnowledgeGraph.build(train)
        dictionary = EntityRecognizer(kg)
        for name, rec in [
            ("dictionary", dictionary),
            ("neural", NeuralRecognizer(ner, kg, dictionary, mode="neural")),
            ("hybrid", NeuralRecognizer(ner, kg, dictionary, mode="hybrid")),
        ]:
            print(f"== {name} on {args.eval}")
            print(format_report(entity_report(docs, rec)))


if __name__ == "__main__":
    main()
