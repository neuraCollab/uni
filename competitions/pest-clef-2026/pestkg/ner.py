"""Entity recognition and linking driven by the knowledge-graph lexicon."""
from __future__ import annotations

import re
from dataclasses import dataclass

from .brat import Entity
from .kg import KnowledgeGraph
from .text import node_key, surface_key, tokens

MONTHS = (r"January|February|March|April|May|June|July|August|September|October|November|December"
          r"|Jan\.?|Feb\.?|Mar\.?|Apr\.?|Jun\.?|Jul\.?|Aug\.?|Sept?\.?|Oct\.?|Nov\.?|Dec\.?")
_NUM = r"(?:\d+|a|one|two|three|four|five|six|seven|eight|nine|ten|twenty|thirty|a few|several|many)"
_UNIT = r"(?:years?|months?|weeks?|days?|decades?|centur(?:y|ies)|hours?|minutes?)"
_DECADES = r"(?:twenties|thirties|forties|fifties|sixties|seventies|eighties|nineties)"
DATE_RE = re.compile(
    r"\b(?:(?:[Ii]n|[Ss]ince|[Ff]rom|[Uu]ntil|[Bb]y|[Dd]uring|[Uu]p to|[Tt]o|[Oo]n|[Aa]s of|[Oo]ver|[Ff]or|[Ww]ithin|[Aa]fter)\s+)?"
    r"(?:(?:early|late|mid|the end of|the beginning of|the|at least|about|around|almost|another)\s+)*"
    r"(?:"
    rf"(?:(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\w*\.?,?\s+)?(?:\d{{1,2}}\s+)?(?:{MONTHS})(?:\s+\d{{1,2}})?(?:,?\s+(?:19|20)\d\d)?"
    r"|(?:19|20)\d\d(?:s|\s*[-–]\s*(?:19|20)?\d\d)?"
    r"|(?:(?:last|this|next|previous|past|coming|recent|same)\s+)+(?:year|years|month|months|season|summer|winter|spring|autumn|decade|week|weekend)"
    r"|(?:this|last|next)?\s*(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)(?:\s+(?:morning|evening|afternoon|night))?"
    rf"|{_NUM}\s+{_UNIT}(?:\s+(?:ago|later|after|earlier|in a row))?"
    rf"|{_DECADES}(?:\s+of the (?:last|past|previous) century)?"
    r"|(?:\w+ )?(?:twentieth|nineteenth|twenty-first|21st|20th) century"
    r"|World War I+"
    r"|so far(?: this year)?|to date|today|for the first time|in recent years|at the moment|currently|recently"
    r"|(?:spring|summer|autumn|winter|winters|summers)"
    r")\b"
)
DISEASE_RE = re.compile(
    r"\b(?:[A-Za-z][\w-]*\s+){0,3}?"
    r"(?:disease|syndrome|wilt|blight|canker(?: stain)?|rot|yellows|yellowing|decline|greening|mosaic|leaf spot|dieback|scab|mildew)\b"
)
_DISEASE_STOP = set("""
the a an this that these those of and or but to in on by with from for at as is are was were be been
its their his her our your which who what when where not no any some all both each every other such same
plant plants pest pests disease diseases new old only main major serious severe dangerous devastating
terrible incurable quarantine live fungal bacterial viral known called named case cases cure spread
incidence management reduction stabilization reports name cycle symptoms present scattered canopy branch
""".split())


def _disease_modifiers(words: list[str]) -> list[str]:
    """Walk back from the head word and keep plausible name modifiers ("olive quick decline")."""
    kept = []
    for w in reversed(words):
        lw = w.lower()
        if lw in _DISEASE_STOP or (w.islower() and (lw.endswith("ing") or lw.endswith("ed") or lw.endswith("ly"))):
            break
        kept.append(w)
    return list(reversed(kept))


ACRONYM_RE = re.compile(r"\s*\(\s*([A-Z][A-Za-z0-9]{1,6})\s*\)")


@dataclass
class Match:
    key: str
    start: int
    end: int
    text: str
    source: str = "lexicon"


class Matcher:
    """Leftmost-longest dictionary matching over tokens, using a token trie built from the KG."""

    def __init__(self, kg: KnowledgeGraph, max_len: int = 8) -> None:
        self.kg = kg
        self.max_len = max_len
        self.keys = set(kg.alias)
        self.prefixes = set()
        for k in self.keys:
            parts = k.split(" ")
            for i in range(1, len(parts)):
                self.prefixes.add(" ".join(parts[:i]))

    def raw_matches(self, text: str) -> list[Match]:
        toks = tokens(text)
        low = [t.lower() for t, _, _ in toks]
        out, i = [], 0
        while i < len(toks):
            best = None
            key = ""
            for j in range(i, min(i + self.max_len, len(toks))):
                key = low[j] if j == i else f"{key} {low[j]}"
                if key in self.keys:
                    best = j
                if key not in self.prefixes:
                    break
            if best is not None:
                k = " ".join(low[i:best + 1])
                s, e = toks[i][1], toks[best][2]
                surface = text[s:e]
                if self._case_ok(k, surface):
                    out.append(Match(k, s, e, surface))
                    i = best + 1
                    continue
            i += 1
        return out

    def _case_ok(self, key: str, surface: str) -> bool:
        if not self.kg.alias_cased.get(key):
            return True
        if surface.isupper() or len(surface) <= 5:
            return surface.isupper() or surface[:1].isupper() and any(c.isupper() for c in surface)
        return surface[:1].isupper()


class EntityRecognizer:
    def __init__(self, kg: KnowledgeGraph, threshold: float = 0.3) -> None:
        self.kg = kg
        self.matcher = Matcher(kg)
        self.threshold = threshold

    # Prior that an external-only match (never seen in training) is a real annotated entity.
    EXTERNAL_PRIOR = {
        "continent": 0.7, "country": 0.7, "ADM1": 0.6, "ADM2": 0.35, "city": 0.35,
        "taxon": 0.6, "genus": 0.2, "common": 0.35,
    }

    def __call__(self, text: str) -> list[Entity]:
        found: list[tuple[int, int, str, str | None, float]] = []
        pending = []
        for m in self.matcher.raw_matches(text):
            cands = self.kg.link_candidates(m.key)
            if not cands:
                continue
            etype, node, count, kind, _ = cands[0]
            prior, seen = self.kg.mention_prior(m.key)
            if seen == 0 and count == 0:  # only known from GeoNames / NCBI
                prior = self.EXTERNAL_PRIOR.get(kind, 0.3)
                if etype == "Location" and kind in ("ADM2", "city"):
                    pending.append((m, cands))
                    continue
            if prior >= self.threshold:
                found.append((m.start, m.end, etype, node, prior))
        found += self._resolve_places(pending, found)
        found += self._dates(text, found)
        found += self._diseases(text, found)
        found += self._acronyms(text, found)
        return self._to_entities(text, found)

    def _resolve_places(self, pending, found):
        """Small places are kept only when the graph puts them inside a place the document
        already mentions (``Limeira`` -> São Paulo -> Brazil), which also disambiguates them."""
        context = {n for _, _, t, n, _ in found if t == "Location"}
        out = []
        for m, cands in pending:
            best = None
            for etype, node, _, kind, prio in cands:
                if etype != "Location":
                    continue
                if context & set(self.kg.ancestors(node)):
                    if best is None or prio > best[1]:
                        best = (node, prio)
            if best:
                out.append((m.start, m.end, "Location", best[0], 0.5))
        return out

    # -- additional recognisers for things the lexicon cannot know ---------------------
    def _dates(self, text, found):
        out = []
        for m in DATE_RE.finditer(text):
            if m.group().strip().lower() in ("the", "may", "a", "for", "in", "march"):
                continue
            if not _overlaps(m.start(), m.end(), found + out):
                out.append((m.start(), m.end(), "Date", None, 0.5))
        return out

    def _diseases(self, text, found):
        """Unseen disease names built from a symptom head word: "olive quick decline syndrome"."""
        out = []
        for m in DISEASE_RE.finditer(text):
            words = m.group().split()
            head_len = 2 if words[-1] in ("stain", "spot") and len(words) > 1 else 1
            words = _disease_modifiers(words[:-head_len]) + words[-head_len:]
            if len(words) <= head_len:
                continue
            start = m.end() - len(" ".join(words))
            if text[start:m.end()] != " ".join(words):
                continue
            if not _overlaps(start, m.end(), found + out):
                out.append((start, m.end(), "Disease", None, 0.4))
        return out

    def _acronyms(self, text, found):
        """"Fall armyworm (FAW)": propagate the long form's type and node to the acronym."""
        out = []
        by_end = {e: (t, n) for s, e, t, n, _ in found}
        defs = {}
        for m in ACRONYM_RE.finditer(text):
            if m.start() in by_end:
                defs[m.group(1)] = by_end[m.start()]
        for acro, (etype, node) in defs.items():
            for m in re.finditer(rf"\b{re.escape(acro)}\b", text):
                if not _overlaps(m.start(), m.end(), found + out):
                    out.append((m.start(), m.end(), etype, node, 0.5))
        return out

    @staticmethod
    def _to_entities(text, found) -> list[Entity]:
        ents = []
        for i, (s, e, etype, node, _) in enumerate(sorted(found), 1):
            norm = node if node and node.split(":", 1)[0] in ("NCBI", "GEO", "OBT") else None
            ents.append(Entity(f"T{i}", etype, [(s, e)], text[s:e], norm))
        return ents


def _overlaps(s, e, spans) -> bool:
    return any(s < se and ss < e for ss, se, *_ in spans)


def entity_node(e: Entity) -> str:
    return node_key(e.type, e.text, e.norm)


__all__ = ["EntityRecognizer", "Matcher", "entity_node", "surface_key"]
