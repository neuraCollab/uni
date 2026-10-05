"""External knowledge graphs: GeoNames (place hierarchy) and the NCBI Taxonomy.

The raw dumps are large, so ``build_cache`` keeps only the names that literally occur in the
corpus texts (a dictionary filter, no labels involved) together with the hierarchy above them,
and stores the result in ``data/ext/external_kg.json``.

    python -m pestkg.resources   # rebuild the cache from dumps already fetched by download_data.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from .text import surface_key, tokens

EXT = Path(__file__).resolve().parent.parent / "data" / "ext"
CACHE = EXT / "external_kg.json"

CONTINENTS = {
    "AF": (6255146, "Africa"), "AS": (6255147, "Asia"), "EU": (6255148, "Europe"),
    "NA": (6255149, "North America"), "OC": (6255151, "Oceania"),
    "SA": (6255150, "South America"), "AN": (6255152, "Antarctica"),
}
# NCBI lineage roots -> entity type proposed for an unseen taxon.
TAXON_ROOTS = [
    (33090, "Plant"),      # Viridiplantae
    (7524, "Vector"),      # Hemiptera: psyllids, leafhoppers, spittlebugs, aphids
    (50557, "Pest"),       # Insecta
    (6231, "Pest"),        # Nematoda
    (4751, "Pest"),        # Fungi
    (4762, "Pest"),        # Oomycota
    (2, "Pest"),           # Bacteria
    (10239, "Pest"),       # Viruses
    (6854, "Pest"),        # Arachnida
]
NAME_CLASSES = {"scientific name": 3.0, "genbank common name": 2.0, "common name": 1.0, "synonym": 0.5}


def corpus_ngrams(texts: list[str], n: int = 7) -> set[str]:
    grams = set()
    for text in texts:
        toks = [t.lower() for t, _, _ in tokens(text)]
        for i in range(len(toks)):
            for j in range(i + 1, min(i + n, len(toks)) + 1):
                grams.add(" ".join(toks[i:j]))
    return grams


def _geonames(grams: set[str]):
    places, parent = [], {}

    def keep(name, gid, kind, prio):
        if surface_key(name) in grams:
            places.append([name, gid, kind, prio])

    for code, (gid, name) in CONTINENTS.items():
        keep(name, gid, "continent", 6.0)
    country_id = {}
    for line in (EXT / "countryInfo.txt").read_text(encoding="utf8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        c = line.split("\t")
        iso, name, cont, gid = c[0], c[4], c[8], int(c[16])
        country_id[iso] = gid
        parent[gid] = CONTINENTS[cont][0] if cont in CONTINENTS else None
        keep(name, gid, "country", 5.0)
    admin1 = {}
    for line in (EXT / "admin1CodesASCII.txt").read_text(encoding="utf8").splitlines():
        code, name, ascii_name, gid = line.split("\t")
        gid = int(gid)
        admin1[code] = gid
        parent[gid] = country_id.get(code.split(".")[0])
        for n in {name, ascii_name}:
            keep(n, gid, "ADM1", 4.0)
    admin2 = {}
    for line in (EXT / "admin2Codes.txt").read_text(encoding="utf8").splitlines():
        code, name, ascii_name, gid = line.split("\t")
        gid = int(gid)
        admin2[code] = gid
        parent[gid] = admin1.get(".".join(code.split(".")[:2]))
        for n in {name, ascii_name}:
            keep(n, gid, "ADM2", 3.0)
    for line in (EXT / "cities5000.txt").open(encoding="utf8"):
        c = line.rstrip("\n").split("\t")
        gid, name, ascii_name, cc, a1, a2, pop = int(c[0]), c[1], c[2], c[8], c[10], c[11], int(c[14] or 0)
        parent[gid] = admin2.get(f"{cc}.{a1}.{a2}") or admin1.get(f"{cc}.{a1}") or country_id.get(cc)
        for n in {name, ascii_name}:
            keep(n, gid, "city", 1.0 + math.log10(pop + 10) / 10)
    # Keep the hierarchy above every kept place.
    needed, frontier = set(), [p[1] for p in places]
    while frontier:
        g = frontier.pop()
        if g in needed or g is None:
            continue
        needed.add(g)
        frontier.append(parent.get(g))
    return places, {g: parent[g] for g in needed if parent.get(g)}


def _ncbi(grams: set[str]):
    names_by_tax: dict[int, list] = {}
    with (EXT / "names.dmp").open(encoding="utf8") as fh:
        for line in fh:
            tax, name, _, cls = (x.strip() for x in line.split("|")[:4])
            prio = NAME_CLASSES.get(cls)
            if prio is None or any(ch.isdigit() for ch in name) or len(name) < 3:
                continue
            if surface_key(name) in grams:
                names_by_tax.setdefault(int(tax), []).append((name, prio))
    parent, rank = {}, {}
    with (EXT / "nodes.dmp").open(encoding="utf8") as fh:
        for line in fh:
            c = line.split("|")
            tax = int(c[0])
            parent[tax] = int(c[1])
            rank[tax] = c[2].strip()
    roots = dict(TAXON_ROOTS)
    taxa = []
    for tax, names in names_by_tax.items():
        etype, node, depth = None, tax, 0
        while node != 1 and depth < 60:
            if node in roots:
                etype = roots[node]
                break
            node, depth = parent.get(node, 1), depth + 1
        if etype is None or rank.get(tax) not in ("species", "genus", "subspecies", "no rank", "strain",
                                                    "varietas", "forma", "family", "serotype"):
            continue
        for name, prio in names:
            # Lower-case common names such as "fig" or "maple" only for plants; they are too
            # ambiguous for other kingdoms ("rust", "blight" ...).
            if prio < 3.0 and etype != "Plant":
                continue
            rank_bonus = 0.5 if rank.get(tax) == "species" else 0.0
            taxa.append([name, tax, etype, prio + rank_bonus])
    return taxa


def build_cache(texts: list[str]) -> dict:
    grams = corpus_ngrams(texts)
    places, geo_parent = _geonames(grams)
    taxa = _ncbi(grams)
    data = {"places": places, "geo_parent": geo_parent, "taxa": taxa}
    CACHE.write_text(json.dumps(data, ensure_ascii=False), encoding="utf8")
    return data


def load_cache() -> dict | None:
    if CACHE.exists():
        return json.loads(CACHE.read_text(encoding="utf8"))
    return None


if __name__ == "__main__":
    docs_dir = EXT.parent / "EPOP_documents"
    texts = [p.read_text(encoding="utf8") for p in docs_dir.glob("*/*.txt")]
    d = build_cache(texts)
    print({k: len(v) for k, v in d.items()})
