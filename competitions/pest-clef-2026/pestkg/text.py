"""Tokenisation, sentence splitting and canonical keys for graph nodes."""
from __future__ import annotations

import re

TOKEN_RE = re.compile(r"\w+|[^\w\s]")
# Sentence end: . ! ? or a newline, but not after a lone capital ("B. dorsalis") or "spp."/"sp."/"cv.".
_SENT_END = re.compile(r"(?<![A-Z])(?<!\bspp)(?<!\bsp)(?<!\bcv)(?<!\bvs)[.!?](?=\s+[A-Z\"(])|\n+")

_LEADING = re.compile(
    r"^(?:the|a|an|in|on|at|since|from|to|until|till|up to|by|during|of|for|between)\s+", re.I
)


def tokens(text: str) -> list[tuple[str, int, int]]:
    return [(m.group(), m.start(), m.end()) for m in TOKEN_RE.finditer(text)]


def sentence_bounds(text: str) -> list[int]:
    """Sorted start offsets of sentences; use with bisect to map a char offset to a sentence index."""
    starts = [0]
    for m in _SENT_END.finditer(text):
        starts.append(m.end())
    return starts


def surface_key(s: str) -> str:
    """Case-folded, whitespace-normalised surface form used for lexicon lookup."""
    return " ".join(t for t, _, _ in tokens(s.lower()))


def text_key(s: str) -> str:
    """Key for entities without an ontology referent (dates, diseases): drop leading
    prepositions/articles so that "in 2021" and "2021" become the same node."""
    k = surface_key(s)
    prev = None
    while prev != k:
        prev, k = k, _LEADING.sub("", k)
    return k


def node_key(etype: str, text: str, norm: str | None) -> str:
    return norm if norm else f"{etype}:{text_key(text)}"
