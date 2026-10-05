"""Domain knowledge graph for plant epidemiomonitoring.

The graph has one node per real-world entity (an NCBI taxon, a GeoNames place, an OntoBiotope
class, or a type-scoped text key for dates and diseases) and three kinds of edges:

* relation edges ``(relation, head, tail)`` aggregated over annotated documents, weighted by the
  number of documents that assert them (Causes, Vected_by, Found_on, Located_in, ...);
* ``part_of`` edges between places (city -> country -> continent) from GeoNames;
* ``alias`` edges from surface forms to nodes, which make up the lexicon used for entity
  recognition and linking.

At inference time the graph is used twice: as a gazetteer to find and link entities, and as a
prior for relation extraction (does the background graph already know ``pest -Vected_by-> X``?
is there a 2-hop path between the two nodes? is the place inside a place known for this pest?).
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from .brat import Document
from .text import node_key, surface_key

# Place names that are mostly ordinary words in this corpus.
PLACE_STOP = {
    "central", "east", "west", "north", "south", "eastern", "western", "northern", "southern", "lot",
    "van", "michael", "patrick", "rot", "draper", "goes", "union", "mobile", "nice", "reading", "bar",
    "victoria", "orange", "lemon", "lime", "banana", "olive", "march",
}


class KnowledgeGraph:
    def __init__(self) -> None:
        self.alias: dict[str, Counter] = defaultdict(Counter)  # surface key -> (type, node) counts
        self.alias_cased: dict[str, bool] = {}  # surface key -> must match case (acronyms, places)
        self.edges: Counter = Counter()  # (rel, head, tail) -> number of documents
        self.out_rel: Counter = Counter()  # (rel, head) -> number of documents
        self.in_rel: Counter = Counter()  # (rel, tail) -> number of documents
        self.adj: dict[str, Counter] = defaultdict(Counter)  # undirected relation adjacency
        self.parent: dict[str, str] = {}  # place -> enclosing place
        self.node_type: dict[str, Counter] = defaultdict(Counter)
        self.label: dict[str, Counter] = defaultdict(Counter)
        # Surface key -> [times annotated, times it occurs in the text] (mention prior).
        self.surface_stats: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        self.genus_type: dict[str, Counter] = defaultdict(Counter)
        self.ext_prio: dict[tuple[str, str, str], tuple[str, float]] = {}  # (key, type, node) -> (kind, prio)

    # ------------------------------------------------------------------ building
    @classmethod
    def build(cls, docs: list[Document], external: bool = True) -> "KnowledgeGraph":
        kg = cls()
        if external:
            kg.add_external()
        for doc in docs:
            kg.add_document(doc)
        kg.count_surface_occurrences(docs)
        return kg

    def add_alias(self, surface: str, etype: str, node: str, weight: int = 1, cased: bool = False) -> None:
        key = surface_key(surface)
        if not key:
            return
        self.alias[key][(etype, node)] += weight
        if cased:
            self.alias_cased.setdefault(key, True)
        else:
            self.alias_cased[key] = False

    def add_external(self) -> None:
        """Merge the GeoNames / NCBI Taxonomy subgraphs (see resources.py) into the graph."""
        from .resources import load_cache

        ext = load_cache()
        if ext is None:
            print("warning: data/ext/external_kg.json missing, run `python -m pestkg.resources`")
            return
        for name, gid, kind, prio in ext["places"]:
            if len(name) < 3 or name.lower() in PLACE_STOP:
                continue
            self.add_external_alias(name, "Location", f"GEO:{gid}", kind, prio)
        for child, par in ext["geo_parent"].items():
            self.parent[f"GEO:{child}"] = f"GEO:{par}"
        for name, tax, etype, prio in ext["taxa"]:
            kind = "common" if prio < 3.0 else "taxon" if " " in name else "genus"
            self.add_external_alias(name, etype, f"NCBI:{tax}", kind, prio)

    def add_external_alias(self, surface: str, etype: str, node: str, kind: str, prio: float) -> None:
        key = surface_key(surface)
        cased = kind != "common"
        self.add_alias(surface, etype, node, weight=0, cased=cased)
        if cased and not surface[:1].isupper():
            self.alias_cased[key] = False
        old = self.ext_prio.get((key, etype, node), (None, -1))[1]
        if prio > old:
            self.ext_prio[(key, etype, node)] = (kind, prio)

    def add_document(self, doc: Document) -> None:
        nodes = {}
        for e in doc.entities.values():
            node = node_key(e.type, e.text, e.norm)
            nodes[e.id] = node
            cased = len(e.text) <= 5 and e.text.isupper()
            self.add_alias(e.text, e.type, node, cased=cased)
            for variant in _variants(e.text, e.type):
                self.add_alias(variant, e.type, node, weight=0, cased=cased)
            self.node_type[node][e.type] += 1
            self.label[node][e.text] += 1
            words = e.text.split()
            if e.type in ("Pest", "Plant", "Vector") and len(words) >= 2 and words[0][:1].isupper() \
                    and len(words[0]) > 2 and words[1].islower():
                self.genus_type[words[0]][e.type] += 1
        seen = set()
        for rel, h, t in doc.relations:
            if h in nodes and t in nodes:
                seen.add((rel, nodes[h], nodes[t]))
        for rel, h, t in seen:
            self.edges[(rel, h, t)] += 1
            self.out_rel[(rel, h)] += 1
            self.in_rel[(rel, t)] += 1
            self.adj[h][t] += 1
            self.adj[t][h] += 1

    def count_surface_occurrences(self, docs: list[Document]) -> None:
        """How often is a known surface form actually annotated when it appears in a text?"""
        from .ner import Matcher

        matcher = Matcher(self)
        for doc in docs:
            gold = {(e.start, e.end) for e in doc.entities.values()}
            gold_starts = {e.start for e in doc.entities.values()}
            for m in matcher.raw_matches(doc.text):
                stats = self.surface_stats[m.key]
                stats[1] += 1
                if (m.start, m.end) in gold or m.start in gold_starts:
                    stats[0] += 1

    # ------------------------------------------------------------------ queries
    def link_candidates(self, key: str) -> list[tuple[str, str, int, str | None, float]]:
        """All (type, node, training count, external kind, external priority), best first."""
        out = []
        for (t, n), c in self.alias.get(key, {}).items():
            kind, prio = self.ext_prio.get((key, t, n), (None, 0.0))
            out.append((t, n, c, kind, prio))
        out.sort(key=lambda x: (-x[2], -x[4], x[0] == "Location"))
        return out

    def link(self, key: str, etype: str | None = None) -> tuple[str, str] | None:
        cands = self.link_candidates(key)
        if etype:
            cands = [c for c in cands if c[0] == etype] or cands
        return (cands[0][0], cands[0][1]) if cands else None

    def mention_prior(self, key: str) -> tuple[float, int]:
        annotated, seen = self.surface_stats.get(key, (0, 0))
        return (annotated + 0.5) / (seen + 1.0), seen

    def ancestors(self, node: str) -> list[str]:
        out = []
        while node in self.parent and len(out) < 6:
            node = self.parent[node]
            out.append(node)
        return out

    def geo_related(self, rel: str, head: str, place: str) -> int:
        """1 if the graph links ``head`` to a place that contains or is contained in ``place``."""
        up = set(self.ancestors(place))
        for (r, h, t), _ in self._edges_from(rel, head):
            if t in up or place in self.ancestors(t):
                return 1
        return 0

    def _edges_from(self, rel: str, head: str):
        if not hasattr(self, "_by_head"):
            self._by_head = defaultdict(list)
            for (r, h, t), c in self.edges.items():
                self._by_head[(r, h)].append(((r, h, t), c))
        return self._by_head.get((rel, head), [])

    def two_hop(self, a: str, b: str) -> int:
        na, nb = self.adj.get(a), self.adj.get(b)
        if not na or not nb:
            return 0
        if len(na) > len(nb):
            na, nb = nb, na
        return sum(1 for x in na if x in nb)

    def degree(self, node: str) -> int:
        return len(self.adj.get(node, ()))

    # ------------------------------------------------------------------ export
    def triples(self):
        """Yield (head, relation, tail, weight) for every edge, including the place hierarchy."""
        for (rel, h, t), c in sorted(self.edges.items()):
            yield h, rel, t, c
        for child, par in sorted(self.parent.items()):
            if child in self.node_type and self.node_type[child].total() > 0:
                yield child, "part_of", par, 1

    def save(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "kg_triples.tsv").open("w", encoding="utf8") as fh:
            fh.write("head\thead_label\trelation\ttail\ttail_label\tdocs\n")
            for h, rel, t, c in self.triples():
                fh.write(f"{h}\t{self.name(h)}\t{rel}\t{t}\t{self.name(t)}\t{c}\n")
        nodes = {
            n: {"types": dict(self.node_type[n]), "labels": dict(self.label[n].most_common(5))}
            for n in self.node_type if self.node_type[n].total() > 0
        }
        (out_dir / "kg_nodes.json").write_text(json.dumps(nodes, ensure_ascii=False, indent=1), encoding="utf8")

    def name(self, node: str) -> str:
        if self.label.get(node):
            return self.label[node].most_common(1)[0][0]
        return node.split(":", 1)[1] if ":" in node and not node[0].isdigit() else node


def _variants(text: str, etype: str) -> list[str]:
    """Cheap morphological and taxonomic variants of an annotated surface form."""
    out = []
    if etype in ("Plant", "Dissemination_pathway", "Disease", "Vector", "Pest"):
        if text.endswith("s") and len(text) > 4:
            out.append(text[:-1])
        elif text[-1:].isalpha():
            out.append(text + "s")
    words = text.split()
    if etype in ("Pest", "Plant", "Vector") and len(words) >= 2 and words[0][:1].isupper() \
            and len(words[0]) > 2 and words[1].islower():
        out.append(f"{words[0][0]}. {' '.join(words[1:])}")  # Xylella fastidiosa -> X. fastidiosa
        out.append(f"{words[0][0]}.{' '.join(words[1:])}")
    return out
