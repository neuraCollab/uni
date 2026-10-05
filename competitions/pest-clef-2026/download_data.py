"""Download the public EPOP corpus and the external knowledge graphs into ./data, then build the
filtered external-KG cache (data/ext/external_kg.json)."""
import io
import tarfile
import urllib.request
import zipfile
from pathlib import Path

DATA = Path(__file__).parent / "data"
EXT = DATA / "ext"

EPOP = {
    # Recherche Data Gouv (Dataverse) file ids.
    "EPOP_documents": "https://entrepot.recherche.data.gouv.fr/api/access/datafile/719158",  # doi:10.57745/YKSEPY
    "EPOP_annotations": "https://entrepot.recherche.data.gouv.fr/api/access/datafile/602246",  # doi:10.57745/ZDNOGF
}
GEONAMES = "https://download.geonames.org/export/dump/"
GEONAMES_FILES = ["countryInfo.txt", "admin1CodesASCII.txt", "admin2Codes.txt", "cities5000.zip"]
NCBI_TAXDUMP = "https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdump.tar.gz"


def fetch(url: str) -> bytes:
    print("downloading", url)
    with urllib.request.urlopen(url) as r:
        return r.read()


if __name__ == "__main__":
    EXT.mkdir(parents=True, exist_ok=True)
    for name, url in EPOP.items():
        if not (DATA / name).exists():
            zipfile.ZipFile(io.BytesIO(fetch(url))).extractall(DATA)
    for name in GEONAMES_FILES:
        target = EXT / name.replace(".zip", ".txt")
        if target.exists():
            continue
        blob = fetch(GEONAMES + name)
        if name.endswith(".zip"):
            zipfile.ZipFile(io.BytesIO(blob)).extractall(EXT)
        else:
            target.write_bytes(blob)
    if not (EXT / "names.dmp").exists():
        with tarfile.open(fileobj=io.BytesIO(fetch(NCBI_TAXDUMP))) as tar:
            for member in ("names.dmp", "nodes.dmp"):
                tar.extract(member, EXT)

    from pestkg.resources import build_cache

    texts = [p.read_text(encoding="utf8") for p in (DATA / "EPOP_documents").glob("*/*.txt")]
    sizes = {k: len(v) for k, v in build_cache(texts).items()}
    print("external KG cache:", sizes)
