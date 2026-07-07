"""Explicit downloader for lattice source files.

This script is never imported by runtime code. It only downloads sources with stable, public file URLs.
Generated packages, API-backed exports, and licensed datasets are reported with instructions instead of
being fetched implicitly.
"""
import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Source:
    url: str | None
    rel_path: str
    license: str
    note: str

    @property
    def downloadable(self) -> bool:
        return bool(self.url)


SOURCES = {
    "onet": Source(
        "https://www.onetcenter.org/dl_files/database/db_30_3_text/Job%20Titles.txt",
        "onet/Job Titles.txt",
        "O*NET database license applies",
        "Current O*NET 30.3 job-title text file.",
    ),
    "cldr-json-full": Source(
        "https://github.com/unicode-org/cldr-json/releases/download/48.2.0/cldr-48.2.0-json-full.zip",
        "nationality/cldr-48.2.0-json-full.zip",
        "Unicode CLDR terms apply",
        "Pinned CLDR JSON full release ZIP.",
    ),
    "esco": Source(
        "https://ec.europa.eu/esco/download/ESCO%20dataset%20-%20v1.2.0%20-%20classification%20-%20%20-%20rdf.zip",
        "profession/esco_v1.2.0_classification_rdf.zip",
        "ESCO data license applies",
        "Direct ESCO v1.2.0 classification RDF package linked by the ESCO portal.",
    ),
    "arda": Source(
        "https://osf.io/download/y9a4j",
        "religion/arda_rcsdem2_stata.dta",
        "ARDA terms apply",
        "RCS-Dem 2.0 completely labeled Stata file.",
    ),
    "disease-ontology": Source(
        "https://raw.githubusercontent.com/DiseaseOntology/HumanDiseaseOntology/main/src/ontology/doid.obo",
        "health/doid.obo",
        "open ontology download",
        "Disease Ontology OBO file.",
    ),
    "mondo": Source(
        "https://raw.githubusercontent.com/monarch-initiative/mondo/master/src/ontology/mondo.obo",
        "health/mondo.obo",
        "open ontology download",
        "Mondo OBO file.",
    ),
    "hancestro": Source(
        "http://purl.obolibrary.org/obo/hancestro.obo",
        "ethnicity/hancestro.obo",
        "open ontology download",
        "Human Ancestry Ontology OBO file.",
    ),
    "gsso": Source(
        "http://purl.obolibrary.org/obo/gsso.owl",
        "categorical/gsso.owl",
        "open ontology download",
        "GSSO OWL file; parser support still needs ontology-specific normalization.",
    ),
    "mesh-rdf": Source(
        "https://id.nlm.nih.gov/mesh/mesh.nt.gz",
        "health/mesh.nt.gz",
        "NLM MeSH terms apply",
        "MeSH RDF N-Triples gzip dump.",
    ),
    "geonames-country-info": Source(
        "https://download.geonames.org/export/dump/countryInfo.txt",
        "nationality/countryInfo.txt",
        "GeoNames data terms apply",
        "Country metadata dump; demonyms require normalization after download.",
    ),
    "fhir-marital-status": Source(
        "https://build.fhir.org/valueset-marital-status.json",
        "categorical/fhir_marital_status.json",
        "FHIR/HL7 terms apply",
        "FHIR CI ValueSet JSON. Pin a published FHIR release for release builds.",
    ),
    "hl7-marital-status": Source(
        "https://terminology.hl7.org/CodeSystem-v3-MaritalStatus.json",
        "categorical/hl7_marital_status.json",
        "HL7 THO terms apply",
        "HL7 Terminology CodeSystem JSON.",
    ),
    "schema-org-person": Source(
        "https://schema.org/docs/jsonldcontext.json",
        "family_role/schema_org_jsonldcontext.json",
        "Schema.org terms apply",
        "Schema.org JSON-LD context. Kinship-property extraction still needs a parser.",
    ),
    "wikidata-lattice-seeds": Source(
        None,
        "wikidata/lattice_seeds.xml",
        "Wikidata CC0; query/export rate limits apply",
        "Targeted SPARQL XML export for occupation/religion/nationality seed labels.",
    ),
    "wikidata": Source(
        None,
        "wikidata/",
        "Wikidata CC0; query/export rate limits apply",
        "Export P106/P140/demonym data offline from dumps or an approved SPARQL job.",
    ),
    "icd11": Source(
        None,
        "health/icd11.jsonl",
        "WHO ICD-11 license and OAuth credentials required",
        "Use a registered ICD API client to export approved local JSONL; do not commit credentials.",
    ),
    "umls": Source(
        None,
        "health/umls.csv",
        "UMLS individual license required",
        "Download/export with an individual UTS account and keep restricted files out of git.",
    ),
}

WIKIDATA_LATTICE_SEEDS_QUERY = """
SELECT ?type ?item ?itemLabel ?alias WHERE {
  {
    VALUES ?item { wd:Q1930187 wd:Q36180 wd:Q40348 wd:Q37226 wd:Q1650915 wd:Q808967 }
    BIND("profession" AS ?type)
  } UNION {
    VALUES ?item { wd:Q9592 wd:Q9268 wd:Q432 wd:Q748 wd:Q23540 wd:Q5043 }
    BIND("religion" AS ?type)
  } UNION {
    VALUES ?item { wd:Q183 wd:Q114 wd:Q145 wd:Q30 wd:Q38 wd:Q142 }
    BIND("nationality" AS ?type)
  }
  OPTIONAL { ?item skos:altLabel ?alias FILTER(LANG(?alias) = "en") }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
LIMIT 500
""".strip()


def _wikidata_url() -> str:
    qs = urllib.parse.urlencode({"query": WIKIDATA_LATTICE_SEEDS_QUERY, "format": "xml"})
    return f"https://query.wikidata.org/sparql?{qs}"


def source_report() -> dict:
    sources = {
        name: {
            "url": _wikidata_url() if name == "wikidata-lattice-seeds" else src.url,
            "path": src.rel_path,
            "license": src.license,
            "note": src.note,
            "downloadable": src.downloadable or name == "wikidata-lattice-seeds",
        }
        for name, src in sorted(SOURCES.items())
    }
    return {
        "downloadable": [name for name, src in sources.items() if src["downloadable"]],
        "manual_or_credentialed": [name for name, src in sources.items() if not src["downloadable"]],
        "sources": sources,
    }


def fetch(name: str, raw_dir: Path, dry_run: bool = False) -> Path:
    src = SOURCES[name]
    out = raw_dir / src.rel_path
    url = _wikidata_url() if name == "wikidata-lattice-seeds" else src.url
    if not url:
        raise SystemExit(f"{name} cannot be downloaded automatically: {src.note}")
    print(f"fetch {name}: {url} -> {out}", flush=True)
    if dry_run:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "agent-cloak-lattice-source-fetcher/0.1"}
    if name == "wikidata-lattice-seeds":
        headers["Accept"] = "application/sparql-results+xml"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            out.write_bytes(r.read())
    except urllib.error.HTTPError as exc:
        if name != "wikidata-lattice-seeds" or exc.code != 429:
            raise
        delay = 65
        print(f"wikidata rate-limited; waiting {delay}s before one retry", flush=True)
        time.sleep(delay)
        with urllib.request.urlopen(req, timeout=120) as r:
            out.write_bytes(r.read())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/lattice_sources/raw")
    ap.add_argument("--source", action="append", choices=sorted(SOURCES))
    ap.add_argument("--all-downloadable", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        print(json.dumps(source_report(), indent=2, sort_keys=True), flush=True)
        return
    selected = list(args.source or [])
    if args.all_downloadable:
        selected.extend(source_report()["downloadable"])
    if not selected:
        raise SystemExit("choose --source, --all-downloadable, or --list")
    for source in dict.fromkeys(selected):
        fetch(source, Path(args.raw_dir), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
