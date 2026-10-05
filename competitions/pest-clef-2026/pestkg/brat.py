"""Reading and writing EPOP documents in the BioNLP-ST (.txt/.a1/.a2) format."""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

# Entity type -> role name it takes in an n-ary event (the corpus uses French role names).
ROLE_OF_TYPE = {
    "Pest": "Organisme_nuisible",
    "Plant": "Plante",
    "Location": "Lieu",
    "Date": "Date",
    "Disease": "Maladie",
    "Vector": "Vecteur",
    "Dissemination_pathway": "Voie_de_dispersion",
}
TYPE_OF_ROLE = {r: t for t, r in ROLE_OF_TYPE.items()}

# Relation type -> (role of the head argument, role of the tail argument).
RELATION_ROLES = {
    "Located_in": ("Object", "Location"),
    "Detected_on": ("Object", "Date"),
    "Found_on": ("Organism", "Habitat"),
    "Causes": ("Pest", "Disease"),
    "Expressed_by": ("Disease", "Plant"),
    "Dispersed_by": ("Nuisance", "Dissemination_pathway"),
    "Vected_by": ("Nuisance", "Vector"),
}
ROLES_RELATION = {roles: rel for rel, roles in RELATION_ROLES.items()}


@dataclass
class Entity:
    id: str
    type: str
    spans: list[tuple[int, int]]
    text: str
    norm: str | None = None  # e.g. "NCBI:7064", "GEO:2658370", "OBT:000289"

    @property
    def start(self) -> int:
        return self.spans[0][0]

    @property
    def end(self) -> int:
        return self.spans[-1][1]


@dataclass
class Document:
    id: str
    text: str
    split: str = ""
    lang: str = ""
    entities: dict[str, Entity] = field(default_factory=dict)
    relations: list[tuple[str, str, str]] = field(default_factory=list)  # (type, head id, tail id)
    events: list[dict[str, str]] = field(default_factory=list)  # role -> entity id


NORM_PREFIX = {"NCBI_Taxonomy": "NCBI", "GeoNames": "GEO", "OntoBiotope": ""}


def _norm_id(kind: str, referent: str) -> str:
    # OntoBiotope referents already look like "OBT:000289".
    prefix = NORM_PREFIX.get(kind, kind)
    return f"{prefix}:{referent}" if prefix else referent


def parse_a2(doc: Document, path: Path) -> None:
    norms = []
    for line in path.read_text(encoding="utf8").splitlines():
        if not line.strip():
            continue
        cols = line.split("\t")
        tag, body = cols[0], cols[1]
        kind = tag[0]
        if kind == "T":
            etype, offsets = body.split(" ", 1)
            spans = [tuple(map(int, s.split())) for s in offsets.split(";")]
            doc.entities[tag] = Entity(tag, etype, spans, cols[2] if len(cols) > 2 else "")
        elif kind == "R":
            rtype, a, b = body.split()
            doc.relations.append((rtype, a.split(":")[1], b.split(":")[1]))
        elif kind == "E":
            args = body.split()[1:]
            doc.events.append({r: e for r, e in (a.split(":", 1) for a in args)})
        elif kind == "N":
            ntype, ann, ref = body.split()
            norms.append((ann.split(":", 1)[1], _norm_id(ntype, ref.split(":", 1)[1])))
    for eid, norm in norms:
        if eid in doc.entities:
            doc.entities[eid].norm = norm


def load_split(docs_dir: Path, ann_dir: Path | None, split: str) -> list[Document]:
    meta = {}
    meta_file = docs_dir / split / "documents-metadata.csv"
    if meta_file.exists():
        with meta_file.open(encoding="utf8") as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                meta[row["DOC"]] = row.get("LANG", "")
    docs = []
    for txt in sorted((docs_dir / split).glob("*.txt")):
        doc = Document(txt.stem, txt.read_text(encoding="utf8"), split, meta.get(txt.stem, ""))
        if ann_dir is not None:
            a2 = ann_dir / split / f"{txt.stem}.a2"
            if a2.exists():
                parse_a2(doc, a2)
        docs.append(doc)
    return docs


def write_a2(doc: Document, path: Path) -> None:
    """Write predicted entities, normalizations, relations and events in BioNLP-ST format."""
    lines, n = [], 0
    for e in doc.entities.values():
        offs = ";".join(f"{s} {t}" for s, t in e.spans)
        lines.append(f"{e.id}\t{e.type} {offs}\t{e.text}")
    for e in doc.entities.values():
        if e.norm:
            n += 1
            prefix, ref = e.norm.split(":", 1)
            kind = {"NCBI": "NCBI_Taxonomy", "GEO": "GeoNames", "OBT": "OntoBiotope"}[prefix]
            ref = e.norm if prefix == "OBT" else ref
            lines.append(f"N{n}\t{kind} Annotation:{e.id} Referent:{ref}")
    for i, (rtype, h, t) in enumerate(doc.relations, 1):
        hr, tr = RELATION_ROLES[rtype]
        lines.append(f"R{i}\t{rtype} {hr}:{h} {tr}:{t}")
    for i, ev in enumerate(doc.events, 1):
        args = " ".join(f"{r}:{e}" for r, e in sorted(ev.items()))
        lines.append(f"E{i}\tNary {args}")
    path.write_text("\n".join(lines) + "\n", encoding="utf8")
