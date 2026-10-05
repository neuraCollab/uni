"""End-to-end model: KG-lexicon NER -> node-level relation classifier -> clique events."""
from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from .brat import RELATION_ROLES, ROLE_OF_TYPE, Document, Entity
from .events import cliques_to_events, event_candidates, gold_events
from .kg import KnowledgeGraph
from .ner import EntityRecognizer
from .relations import RELATIONS, build_matrix, gold_triples


@dataclass
class Prediction:
    doc_id: str
    entities: list[Entity]
    node_type: dict[str, str]
    scores: dict[tuple[str, str, str], float]
    triples: set[tuple[str, str, str]] = field(default_factory=set)
    events: list[frozenset] = field(default_factory=list)
    dg: object = None


class PestKGModel:
    def __init__(self, folds: int = 5, ner_threshold: float = 0.3, threshold: float = 0.4,
                 gold_entities: bool = False, augment_gold: bool = True, tune: bool = True,
                 seed: int = 13) -> None:
        self.folds = folds
        self.ner_threshold = ner_threshold
        self.threshold = threshold  # global; per-relation overrides in self.rel_threshold
        self.rel_threshold: dict[str, float] = {}
        self.gold_entities = gold_entities
        self.augment_gold = augment_gold
        self.tune = tune
        self.seed = seed

    def _entities(self, doc: Document, ner: EntityRecognizer) -> list[Entity]:
        return list(doc.entities.values()) if self.gold_entities else ner(doc.text)

    def _classifier(self):
        return HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=20,
            l2_regularization=1.0, categorical_features=[0, 1, 2], random_state=self.seed,
        )

    def fit(self, docs: list[Document]) -> "PestKGModel":
        """Cross-fitting: features for each training document come from a KG and lexicon built
        without that document, so the classifier sees KG evidence as noisy as at test time.
        The same folds give out-of-fold scores used to pick one threshold per relation."""
        order = docs[:]
        random.Random(self.seed).shuffle(order)
        X, y, fold, aug, records = [], [], [], [], []
        missed = Counter()  # gold triples whose nodes the NER did not find (unrecoverable FN)
        for k in range(self.folds):
            held = order[k::self.folds]
            held_ids = {d.id for d in held}
            kg = KnowledgeGraph.build([d for d in order if d.id not in held_ids])
            ner = EntityRecognizer(kg, self.ner_threshold)
            for doc in held:
                dg, keys, Xd, yd = build_matrix(doc, self._entities(doc, ner), kg)
                records.append((doc, dg, keys, k))
                found = {kk for kk, yy in zip(keys, yd) if yy}
                for rel, *_ in gold_triples(doc) - found:
                    missed[rel] += 1
                if len(Xd):
                    X.append(Xd); y.append(yd); fold.append(np.full(len(yd), k))
                if self.augment_gold and not self.gold_entities:
                    # Same document seen through gold entities: many more positive examples.
                    _, _, Xg, yg = build_matrix(doc, list(doc.entities.values()), kg)
                    if len(Xg):
                        aug.append((Xg, yg, k))
        X, y, fold = np.vstack(X), np.concatenate(y), np.concatenate(fold)
        Xa = np.vstack([a[0] for a in aug]) if aug else np.empty((0, X.shape[1]))
        ya = np.concatenate([a[1] for a in aug]) if aug else np.empty(0)
        fa = np.concatenate([np.full(len(a[1]), a[2]) for a in aug]) if aug else np.empty(0)

        if self.tune:
            oof = np.zeros(len(y))
            for k in range(self.folds):
                tr, te = fold != k, fold == k
                clf = self._classifier().fit(np.vstack([X[tr], Xa[fa != k]]), np.concatenate([y[tr], ya[fa != k]]))
                oof[te] = clf.predict_proba(X[te])[:, 1]
            self.rel_threshold = _tune(X[:, 0], y, oof, missed)
            self._fit_events(records, oof)
        self.clf = self._classifier().fit(np.vstack([X, Xa]), np.concatenate([y, ya]))
        self.kg = KnowledgeGraph.build(docs)
        self.ner = EntityRecognizer(self.kg, self.ner_threshold)
        return self

    def _fit_events(self, records, oof) -> None:
        """Learn which cliques of the (out-of-fold) relation graph are real n-ary events."""
        Xe, ye, fe, n_gold = [], [], [], 0
        i = 0
        for doc, dg, keys, k in records:
            scores = dict(zip(keys, oof[i:i + len(keys)]))
            i += len(keys)
            gold = gold_events(doc)
            n_gold += sum(1 for ev in gold if len(ev) > 1)
            for ev, feats in event_candidates(scores, dg.type, dg, self.rel_threshold):
                Xe.append(feats); ye.append(int(ev in gold)); fe.append(k)
        Xe, ye, fe = np.array(Xe, dtype=float), np.array(ye), np.array(fe)
        oof_e = np.zeros(len(ye))
        for k in range(self.folds):
            clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=self.seed)
            oof_e[fe == k] = clf.fit(Xe[fe != k], ye[fe != k]).predict_proba(Xe[fe == k])[:, 1]
        best = (0.0, 0.5)
        for t in np.arange(0.05, 0.91, 0.025):
            tp = ((oof_e >= t) & (ye == 1)).sum()
            best = max(best, (2 * tp / ((oof_e >= t).sum() + n_gold), -t))
        self.event_threshold = float(-best[1])
        self.event_clf = HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, random_state=self.seed).fit(Xe, ye)

    def predict(self, docs: list[Document]) -> list[Prediction]:
        preds = []
        for doc in docs:
            ents = self._entities(doc, self.ner)
            dg, keys, X, _ = build_matrix(doc, ents, self.kg, with_labels=False)
            proba = self.clf.predict_proba(X)[:, 1] if len(X) else []
            scores = dict(zip(keys, map(float, proba)))
            preds.append(Prediction(doc.id, ents, dg.type, scores, dg=dg))
        self.decide(preds)
        return preds

    def decide(self, preds: list[Prediction]) -> None:
        for p in preds:
            p.triples = {k for k, s in p.scores.items() if s >= self.rel_threshold.get(k[0], self.threshold)}
            if getattr(self, "event_clf", None) is None:
                p.events = cliques_to_events(p.triples, p.node_type)
                continue
            th = {r: self.rel_threshold.get(r, self.threshold) for r in RELATIONS}
            cands = event_candidates(p.scores, p.node_type, p.dg, th)
            if not cands:
                p.events = []
                continue
            proba = self.event_clf.predict_proba(np.array([f for _, f in cands], dtype=float))[:, 1]
            p.events = sorted((ev for (ev, _), s in zip(cands, proba) if s >= self.event_threshold), key=sorted)


def _tune(rel_idx, y, scores, missed) -> dict[str, float]:
    """Per-relation threshold maximising out-of-fold F1 (counting NER misses as false negatives)."""
    out = {}
    for i, rel in enumerate(RELATIONS):
        m = rel_idx == i
        yr, sr = y[m], scores[m]
        n_gold = yr.sum() + missed[rel]
        best = (0.0, 0.5)
        for t in np.arange(0.05, 0.91, 0.025):
            pred = sr >= t
            tp = (pred & (yr == 1)).sum()
            f1 = 2 * tp / (pred.sum() + n_gold) if pred.sum() + n_gold else 0.0
            best = max(best, (f1, -t))
        out[rel] = round(float(-best[1]), 3)
    return out


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
        nodes = dict(ev)
        anchor = None
        args = {}
        for role, node in sorted(nodes.items()):
            ms = dg.mentions[node]
            m = ms[0] if anchor is None else min(ms, key=lambda x: abs(x.start - anchor))
            anchor = m.start if anchor is None else anchor
            args[role] = m.id
        out.events.append(args)
    return out


__all__ = ["PestKGModel", "Prediction", "to_document", "RELATIONS", "RELATION_ROLES", "ROLE_OF_TYPE"]
