"""Transformer NER (BIO token classification) + linking of the predicted spans to the KG.

The neural model only decides *where* the entities are and of which type; *which* node of the
knowledge graph they denote (NCBI taxon, GeoNames place, ...) is still resolved by the graph:
lexicon lookup, morphological/taxonomic variants, place hierarchy in the document context and,
as a last resort, fuzzy matching against the aliases of that type.
"""
from __future__ import annotations

import math
import random
import re
import time
from pathlib import Path

import numpy as np

from .brat import Document, Entity
from .kg import KnowledgeGraph
from .text import surface_key, text_key

TYPES = ["Pest", "Plant", "Location", "Date", "Disease", "Vector", "Dissemination_pathway"]
LABELS = ["O"] + [f"{p}-{t}" for t in TYPES for p in ("B", "I")]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
WORD_RE = re.compile(r"\w+|[^\w\s]")


def words_of(text: str) -> list[tuple[str, int, int]]:
    return [(m.group(), m.start(), m.end()) for m in WORD_RE.finditer(text)]


def bio_labels(doc: Document, words) -> list[int]:
    """Word-level BIO labels; discontinuous entities use their first fragment, and when
    annotations overlap the longer one wins."""
    labels = ["O"] * len(words)
    spans = sorted(((e.spans[0][0], e.spans[0][1], e.type) for e in doc.entities.values()),
                   key=lambda s: -(s[1] - s[0]))
    taken = [False] * len(words)
    for s, e, t in spans:
        idx = [i for i, (_, ws, we) in enumerate(words) if ws >= s and we <= e]
        if not idx or any(taken[i] for i in idx):
            continue
        for k, i in enumerate(idx):
            labels[i] = ("B-" if k == 0 else "I-") + t
            taken[i] = True
    return [LABEL2ID[l] for l in labels]


class NeuralNER:
    def __init__(self, model_dir_or_name: str, device: str | None = None, max_length: int = 512,
                 stride: int = 128) -> None:
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_dir_or_name)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_dir_or_name, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)), label2id=LABEL2ID,
            ignore_mismatched_sizes=True,
        ).to(self.device)
        self.max_length = max_length
        self.stride = stride

    # ------------------------------------------------------------------ data
    def _chunks(self, words: list[str], labels: list[int] | None = None):
        enc = self.tok(words, is_split_into_words=True, truncation=True, max_length=self.max_length,
                       stride=self.stride, return_overflowing_tokens=True)
        out = []
        for i in range(len(enc["input_ids"])):
            wids = enc.word_ids(i)
            lab, prev = [], None
            for w in wids:
                if w is None or w == prev:
                    lab.append(-100)
                else:
                    lab.append(labels[w] if labels is not None else 0)
                prev = w
            out.append({"input_ids": enc["input_ids"][i], "attention_mask": enc["attention_mask"][i],
                        "labels": lab, "word_ids": wids})
        return out

    def _batch(self, chunks):
        n = max(len(c["input_ids"]) for c in chunks)
        pad = self.tok.pad_token_id or 0
        t = self.torch
        ids = t.tensor([c["input_ids"] + [pad] * (n - len(c["input_ids"])) for c in chunks])
        att = t.tensor([c["attention_mask"] + [0] * (n - len(c["attention_mask"])) for c in chunks])
        lab = t.tensor([c["labels"] + [-100] * (n - len(c["labels"])) for c in chunks])
        return ids.to(self.device), att.to(self.device), lab.to(self.device)

    # ------------------------------------------------------------------ training
    def fit(self, docs: list[Document], epochs: int = 8, lr: float = 5e-5, batch_size: int = 8,
            warmup: float = 0.1, seed: int = 13, log=print) -> "NeuralNER":
        from transformers import get_linear_schedule_with_warmup

        t = self.torch
        t.manual_seed(seed)
        random.seed(seed)
        chunks = []
        for doc in docs:
            ws = words_of(doc.text)
            chunks += self._chunks([w for w, _, _ in ws], bio_labels(doc, ws))
        steps = epochs * math.ceil(len(chunks) / batch_size)
        opt = t.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        sched = get_linear_schedule_with_warmup(opt, int(warmup * steps), steps)
        use_amp = self.device == "cuda"
        scaler = t.amp.GradScaler("cuda") if use_amp else None
        self.model.train()
        for ep in range(epochs):
            random.shuffle(chunks)
            total, t0 = 0.0, time.time()
            for i in range(0, len(chunks), batch_size):
                ids, att, lab = self._batch(chunks[i:i + batch_size])
                with t.autocast(device_type="cuda", dtype=t.float16, enabled=use_amp):
                    loss = self.model(input_ids=ids, attention_mask=att, labels=lab).loss
                opt.zero_grad()
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    t.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    t.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    opt.step()
                sched.step()
                total += loss.item() * len(ids)
            log(f"epoch {ep + 1}/{epochs}: loss {total / len(chunks):.4f} ({time.time() - t0:.0f}s, {len(chunks)} chunks)")
        self.model.eval()
        return self

    def save(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(out_dir)
        self.tok.save_pretrained(out_dir)

    # ------------------------------------------------------------------ inference
    def spans(self, text: str, batch_size: int = 8) -> list[tuple[int, int, str, float]]:
        """(start, end, type, confidence) of predicted entities in ``text``."""
        t = self.torch
        ws = words_of(text)
        if not ws:
            return []
        chunks = self._chunks([w for w, _, _ in ws])
        probs = np.zeros((len(ws), len(LABELS)))
        counts = np.zeros(len(ws))
        self.model.eval()
        with t.no_grad():
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                ids, att, _ = self._batch(batch)
                p = t.softmax(self.model(input_ids=ids, attention_mask=att).logits.float(), -1).cpu().numpy()
                for b, c in enumerate(batch):
                    prev = None
                    for j, w in enumerate(c["word_ids"]):
                        if w is not None and w != prev:
                            probs[w] += p[b, j]
                            counts[w] += 1
                        prev = w
        probs /= np.maximum(counts, 1)[:, None]
        pred = probs.argmax(-1)
        out, cur = [], None
        for i, li in enumerate(pred):
            lab = LABELS[li]
            if lab == "O":
                if cur:
                    out.append(cur)
                cur = None
                continue
            bio, typ = lab.split("-", 1)
            if cur and bio == "I" and cur[2] == typ:
                cur = (cur[0], ws[i][2], typ, min(cur[3], probs[i, li]))
            else:
                if cur:
                    out.append(cur)
                cur = (ws[i][1], ws[i][2], typ, float(probs[i, li]))
        if cur:
            out.append(cur)
        return out


# ---------------------------------------------------------------------- linking
_TRAIL_PAREN = re.compile(r"\s*\([^)]*\)\s*$")
_HEAD_DROP = {"plants", "plant", "trees", "tree", "crops", "crop", "fruits", "fruit", "orchards", "orchard",
              "groves", "grove", "seedlings", "leaves", "virus", "bacterium", "fungus", "insect", "nematode",
              "region", "regions", "province", "area", "canton", "cantons", "municipality", "state",
              "department", "county", "district", "island", "islands", "city"}


class Linker:
    """Map a (surface, type) pair onto a knowledge-graph node."""

    def __init__(self, kg: KnowledgeGraph, fuzzy: int = 90) -> None:
        self.kg = kg
        self.fuzzy = fuzzy
        self.by_type: dict[str, list[str]] = {}
        for key, cands in kg.alias.items():
            for (t, _n) in cands:
                self.by_type.setdefault(t, []).append(key)

    def candidates_for(self, surface: str):
        """Surface variants, most specific first."""
        s = surface.strip()
        yield s
        yield _TRAIL_PAREN.sub("", s)
        k = text_key(s)
        yield k
        words = k.split()
        if words and words[-1] in _HEAD_DROP and len(words) > 1:
            yield " ".join(words[:-1])
        if words and words[0] in _HEAD_DROP and len(words) > 1:
            yield " ".join(w for w in words[1:] if w != "of")
        if k.endswith("s"):
            yield k[:-1]
        else:
            yield k + "s"
        if len(words) > 1:
            yield words[-1]

    def link(self, surface: str, etype: str, context: set[str]) -> str | None:
        compatible = {etype} | ({"Pest", "Vector"} if etype in ("Pest", "Vector") else set())
        for variant in self.candidates_for(surface):
            cands = [c for c in self.kg.link_candidates(surface_key(variant)) if c[0] in compatible]
            if not cands:
                continue
            if etype == "Location" and len(cands) > 1:
                inside = [c for c in cands if context & set(self.kg.ancestors(c[1]))]
                if inside and inside[0][2] == 0:  # no training evidence: trust the hierarchy
                    return inside[0][1]
            return cands[0][1]
        if self.fuzzy and etype in self.by_type:
            from rapidfuzz import fuzz, process

            hit = process.extractOne(surface_key(surface), self.by_type[etype], scorer=fuzz.ratio,
                                     score_cutoff=self.fuzzy)
            if hit:
                cands = [c for c in self.kg.link_candidates(hit[0]) if c[0] == etype]
                if cands:
                    return cands[0][1]
        return None


class NeuralRecognizer:
    """Drop-in replacement for ``EntityRecognizer``: neural spans, KG linking, and optionally
    the dictionary recogniser's entities where the network found nothing (``mode='hybrid'``)."""

    def __init__(self, ner: NeuralNER, kg: KnowledgeGraph, dictionary=None, mode: str = "neural") -> None:
        self.ner = ner
        self.kg = kg
        self.linker = Linker(kg)
        self.dictionary = dictionary
        self.mode = mode

    def __call__(self, text: str) -> list[Entity]:
        spans = self.ner.spans(text)
        dict_ents = self.dictionary(text) if self.dictionary is not None else []
        # Places the dictionary is sure about give the context for disambiguating the others.
        context = {e.norm for e in dict_ents if e.type == "Location" and e.norm}
        found = []
        for s, e, typ, conf in spans:
            node = self.linker.link(text[s:e], typ, context)
            found.append((s, e, typ, node, conf))
            if typ == "Location" and node:
                context.add(node)
        if self.mode == "hybrid":
            for d in dict_ents:
                if not any(d.start < e and s < d.end for s, e, *_ in found):
                    found.append((d.start, d.end, d.type, d.norm, 0.5))
        ents = []
        for i, (s, e, typ, node, _) in enumerate(sorted(found), 1):
            norm = node if node and node.split(":", 1)[0] in ("NCBI", "GEO", "OBT") else None
            ents.append(Entity(f"T{i}", typ, [(s, e)], text[s:e], norm))
        return ents
