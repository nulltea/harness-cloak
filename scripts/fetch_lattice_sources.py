"""Explicit downloader for open lattice source files.

This script is never imported by runtime code and should be run only when the user explicitly wants to
refresh local source artifacts.
"""
import argparse
import urllib.request
from pathlib import Path

SOURCES = {
    "disease-ontology": (
        "https://raw.githubusercontent.com/DiseaseOntology/HumanDiseaseOntology/main/src/ontology/doid.obo",
        "health/doid.obo",
    ),
    "mondo": (
        "https://raw.githubusercontent.com/monarch-initiative/mondo/master/src/ontology/mondo.obo",
        "health/mondo.obo",
    ),
}


def fetch(name: str, raw_dir: Path) -> Path:
    url, rel = SOURCES[name]
    out = raw_dir / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"fetch {name}: {url} -> {out}", flush=True)
    urllib.request.urlretrieve(url, out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/lattice_sources/raw")
    ap.add_argument("--source", action="append", choices=sorted(SOURCES), required=True)
    args = ap.parse_args()
    for source in args.source:
        fetch(source, Path(args.raw_dir))


if __name__ == "__main__":
    main()
