"""Build fine runtime lattice profiles from cached raw datasets.

Network access is deliberately absent here. Use scripts/download/fetch_lattice_sources.py or manual downloads
to populate data/lattice_sources/raw first.
"""
import argparse
import json
import re
import zipfile
from collections import defaultdict
from datetime import date
from pathlib import Path

from cloak.lattice.profiles import SCHEMA_VERSION, validate_profile_artifact
from cloak.runtime_types import PLACEHOLDER_ONLY_TYPES
from lattice_sources.categorical import alias_rows
from lattice_sources.common import ProfileRow, norm
from lattice_sources.demographics import rows_from_cldr_zip, rows_from_wikidata_sparql_xml
from lattice_sources.drugs import rows_from_openfda_ndc_zip
from lattice_sources.geonames import rows_from_geonames
from lattice_sources.legacy_cache import rows_from_legacy_teacher_cache
from lattice_sources.obo import rows_from_obo
from lattice_sources.occupation import rows_from_esco_rdf, rows_from_isco_csv, rows_from_onet_job_titles, rows_from_onet_titles
from lattice_sources.organizations import rows_from_nppes_zip
from lattice_sources.procedures import rows_from_icd10_pcs_order_zip
from lattice_sources.religion import rows_from_arda_stata

HEALTH_FAMILY_ROOTS = {
    "DOID:0014667": "metabolic condition",
    "DOID:0050117": "infectious disease",
    "DOID:0060118": "thoracic condition",
    "DOID:15": "reproductive system condition",
    "DOID:16": "skin condition",
    "DOID:17": "musculoskeletal condition",
    "DOID:18": "urinary system condition",
    "DOID:28": "endocrine condition",
    "DOID:74": "hematologic condition",
    "DOID:77": "gastrointestinal condition",
    "DOID:150": "mental health condition",
    "DOID:225": "syndrome",
    "DOID:863": "neurological condition",
    "DOID:1287": "cardiovascular condition",
    "DOID:14566": "neoplastic condition",
    "DOID:1579": "respiratory condition",
    "DOID:2914": "immune system condition",
    "MONDO:0005151": "infectious disease",
    "MONDO:0005084": "respiratory condition",
    "MONDO:0002025": "mental health condition",
}

CATEGORICAL_ALIASES = {
    "gender": {"female": ["woman"], "male": ["man"]},
    "marital-status": {"married": ["wedded"], "divorced": ["formerly married"], "widowed": ["widow", "widower"]},
    "sexual-orientation": {"gay": ["homosexual"], "bisexual": ["bi"], "lesbian": []},
}

NON_INFORMATIVE_LEVELS = {
    ("profession", ("worker",)),
}


def _surface_allowed(runtime_type: str, surface: str) -> bool:
    if runtime_type == "profession" and len(norm(surface).split()) > 2:
        return False
    if runtime_type == "health-condition":
        surface = norm(surface)
        if "," in surface or len(surface.split()) > 3 or re.search(r"\b\d+$", surface):
            return False
    return True


def _informative_levels(runtime_type: str, surface: str, levels: list[str]) -> list[str]:
    surface = norm(surface)
    out = []
    for level in levels:
        level = norm(level)
        if surface and surface in level:
            continue
        if level and level not in out:
            out.append(level)
    if (runtime_type, tuple(out)) in NON_INFORMATIVE_LEVELS:
        return []
    return out


def _add_row(dst: dict, row: ProfileRow) -> None:
    rt = row.runtime_type
    surface = norm(row.surface)
    if not surface or not _surface_allowed(rt, surface):
        return
    levels = _informative_levels(rt, surface, row.levels)
    if row.levels and not levels:
        return
    cur = dst[rt].setdefault(surface, {"aliases": set(), "levels": [], "source_ids": set(), "count": 1.0})
    cur["aliases"].update(a for a in row.aliases if norm(a) and norm(a) != surface)
    for level in levels:
        if level not in cur["levels"]:
            cur["levels"].append(level)
    cur["source_ids"].update(s for s in row.source_ids if s)
    cur["count"] = max(float(cur["count"]), float(row.count))


def merge_rows(rows: list[ProfileRow]) -> dict:
    merged = defaultdict(dict)
    for row in rows:
        _add_row(merged, row)
    profiles = {}
    for rt, entries in sorted(merged.items()):
        profiles[rt] = {}
        for surface, row in sorted(entries.items()):
            profiles[rt][surface] = {
                "aliases": sorted(row["aliases"]),
                "levels": row["levels"],
                "source_ids": sorted(row["source_ids"]),
                "count": row["count"],
            }
    return {
        "schema_version": SCHEMA_VERSION,
        "created": str(date.today()),
        "sources": {},
        "profiles": profiles,
    }


def collect_rows(
    raw_dir: Path,
    geo_dir: Path | None = None,
    teacher_cache: Path | None = None,
) -> list[ProfileRow]:
    rows = []
    if geo_dir and geo_dir.exists():
        rows.extend(rows_from_geonames(geo_dir))
    if teacher_cache and teacher_cache.exists():
        rows.extend(rows_from_legacy_teacher_cache(teacher_cache))
    onet = raw_dir / "onet" / "Alternate Titles.txt"
    if onet.exists():
        rows.extend(rows_from_onet_titles(onet))
    onet_jobs = raw_dir / "onet" / "Job Titles.txt"
    if onet_jobs.exists():
        rows.extend(rows_from_onet_job_titles(onet_jobs))
    isco = raw_dir / "isco" / "isco.csv"
    if isco.exists():
        rows.extend(rows_from_isco_csv(isco))
    esco = raw_dir / "profession" / "esco_v1.2.0_classification_rdf.zip"
    if esco.exists():
        with zipfile.ZipFile(esco) as zf, zf.open("esco-v1.2.0.rdf") as rdf:
            rows.extend(rows_from_esco_rdf(rdf))
    arda = raw_dir / "religion" / "arda_rcsdem2_stata.dta"
    if arda.exists():
        rows.extend(rows_from_arda_stata(arda))
    cldr = raw_dir / "nationality" / "cldr-48.2.0-json-full.zip"
    if cldr.exists():
        rows.extend(rows_from_cldr_zip(cldr))
    wikidata = raw_dir / "wikidata" / "lattice_seeds.xml"
    if wikidata.exists():
        rows.extend(rows_from_wikidata_sparql_xml(wikidata))
    openfda_ndc = raw_dir / "drug" / "openfda_ndc.json.zip"
    if openfda_ndc.exists():
        rows.extend(rows_from_openfda_ndc_zip(openfda_ndc))
    icd10pcs = raw_dir / "procedure" / "icd10pcs_order_2026.zip"
    if icd10pcs.exists():
        rows.extend(rows_from_icd10_pcs_order_zip(icd10pcs))
    nppes = raw_dir / "org" / "nppes_weekly_v2.zip"
    if nppes.exists():
        rows.extend(rows_from_nppes_zip(nppes))
    for name in ("mondo.obo", "doid.obo"):
        path = raw_dir / "health" / name
        if path.exists():
            rows.extend(rows_from_obo(path, "health-condition", HEALTH_FAMILY_ROOTS))
    for runtime_type, aliases in CATEGORICAL_ALIASES.items():
        rows.extend(alias_rows(runtime_type, aliases))
    return rows


def coverage_report(artifact: dict) -> dict:
    profiles = artifact.get("profiles", {})
    return {
        "profile_counts": {runtime_type: len(rows) for runtime_type, rows in sorted(profiles.items())},
        "placeholder_only_types": sorted(PLACEHOLDER_ONLY_TYPES),
        "placeholder_first_types": ["demographic-other"],
        "notes": [
            "Coverage counts candidate surfaces only; they are not privacy guarantees.",
            "Every emitted text level still requires lattice filters and anonymity floors.",
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/lattice_sources/raw")
    ap.add_argument("--geo-dir", default="data/geonames")
    ap.add_argument("--teacher-cache", default="data/lattice_cache.json")
    ap.add_argument("--out", default="data/lattice_profiles/fine_lattice_profiles.json")
    ap.add_argument("--coverage-out", default=None)
    args = ap.parse_args()
    rows = collect_rows(Path(args.raw_dir), Path(args.geo_dir), Path(args.teacher_cache))
    artifact = merge_rows(rows)
    errors = validate_profile_artifact(artifact)
    if errors:
        raise SystemExit("invalid lattice profile artifact:\n" + "\n".join(errors[:50]))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, sort_keys=True))
    print(f"rows={len(rows)} profiles={sum(len(v) for v in artifact['profiles'].values())} -> {out}", flush=True)
    if args.coverage_out:
        cov = Path(args.coverage_out)
        cov.parent.mkdir(parents=True, exist_ok=True)
        cov.write_text(json.dumps(coverage_report(artifact), indent=2, sort_keys=True))
        print(f"coverage -> {cov}", flush=True)


if __name__ == "__main__":
    main()
