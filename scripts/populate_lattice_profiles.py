"""Populate fine lattice profiles from local cached source files.

This is an offline orchestration script. It does not download source data or call remote APIs; it reports
which exhaustive-target sources are absent or still need parser support.
"""
import argparse
import json
import sys
from pathlib import Path

from build_lattice_profiles import collect_rows, coverage_report, merge_rows
from cloak.lattice_profiles import validate_profile_artifact

SUPPORTED_SOURCES = {
    "onet-job-titles": {
        "runtime_types": ["profession"],
        "paths": ["onet/Job Titles.txt"],
        "required": True,
    },
    "onet-alternate-titles": {
        "runtime_types": ["profession"],
        "paths": ["onet/Alternate Titles.txt"],
        "required": False,
    },
    "isco08": {
        "runtime_types": ["profession"],
        "paths": ["isco/isco.csv"],
        "required": False,
    },
    "disease-ontology": {
        "runtime_types": ["health-condition"],
        "paths": ["health/doid.obo"],
        "required": False,
    },
    "mondo": {
        "runtime_types": ["health-condition"],
        "paths": ["health/mondo.obo"],
        "required": False,
    },
    "esco-rdf": {
        "runtime_types": ["profession"],
        "paths": ["profession/esco_v1.2.0_classification_rdf.zip"],
        "required": True,
    },
    "arda": {
        "runtime_types": ["religion"],
        "paths": ["religion/arda_rcsdem2_stata.dta"],
        "required": True,
    },
    "cldr-territories": {
        "runtime_types": ["nationality"],
        "paths": ["nationality/cldr-48.2.0-json-full.zip"],
        "required": True,
    },
    "wikidata-lattice-seeds": {
        "runtime_types": ["profession", "religion", "nationality"],
        "paths": ["wikidata/lattice_seeds.xml"],
        "required": True,
    },
    "manual-categorical-aliases": {
        "runtime_types": ["gender", "marital-status", "sexual-orientation"],
        "paths": [],
        "required": True,
    },
    "openfda-ndc": {
        "runtime_types": ["drug"],
        "paths": ["drug/openfda_ndc.json.zip"],
        "required": False,
    },
    "icd10-pcs-order": {
        "runtime_types": ["medical-procedure"],
        "paths": ["procedure/icd10pcs_order_2026.zip"],
        "required": False,
    },
    "nppes-weekly": {
        "runtime_types": ["organization-medical-facility"],
        "paths": ["org/nppes_weekly_v2.zip"],
        "required": False,
    },
}

UNIMPLEMENTED_SOURCES = {
    "mesh-rdf": ["health-condition"],
    "icd11": ["health-condition"],
    "umls": ["health-condition"],
    "hancestro": ["ethnicity", "nationality"],
    "census-race-ethnicity-standards": ["ethnicity"],
    "geonames-demonyms": ["nationality"],
    "kin-ontology": ["family-role"],
    "schema-org-person-relations": ["family-role"],
    "mesh-age-groups": ["age"],
    "cdc-age-groups": ["age"],
    "gsso": ["gender", "sexual-orientation"],
    "fhir-marital-status": ["marital-status"],
    "hl7-marital-status": ["marital-status"],
}


def _present(raw_dir: Path, rel_paths: list[str]) -> bool:
    return all((raw_dir / rel).exists() for rel in rel_paths)


def population_report(
    raw_dir: Path,
    artifact: dict,
    geo_dir: Path | None = None,
    teacher_cache: Path | None = None,
) -> dict:
    available = []
    missing = []
    blocking_missing = []
    for name, spec in sorted(SUPPORTED_SOURCES.items()):
        if _present(raw_dir, spec["paths"]):
            available.append(name)
        else:
            missing.append(name)
            if spec.get("required"):
                blocking_missing.append(name)
    if geo_dir and (geo_dir / "countryInfo.txt").exists():
        available.append("geonames-loc")
    else:
        missing.append("geonames-loc")
    if teacher_cache and teacher_cache.exists():
        available.append("legacy-teacher-cache")
    else:
        missing.append("legacy-teacher-cache")
    unsupported = sorted(UNIMPLEMENTED_SOURCES)
    return {
        "raw_dir": str(raw_dir),
        "exhaustive": not blocking_missing,
        "available_sources": available,
        "missing_sources": missing,
        "blocking_missing_sources": blocking_missing,
        "optional_missing_sources": [name for name in missing if name not in set(blocking_missing)],
        "unimplemented_sources": unsupported,
        "unimplemented_source_types": UNIMPLEMENTED_SOURCES,
        "profile_counts": coverage_report(artifact)["profile_counts"],
        "notes": [
            "Exhaustive means all required downloaded-source families have local files and parser support.",
            "This report describes candidate lattice-source coverage, not privacy protection.",
            "The script is offline-only and never downloads data.",
            "Use --require-exhaustive to fail while any required downloaded-source family is missing.",
            "GeoNames LOC and legacy teacher cache are optional local cache inputs and are reported when present.",
            "Unimplemented sources are future/manual expansion families; they do not block the downloaded-source build.",
        ],
    }


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/lattice_sources/raw")
    ap.add_argument("--geo-dir", default="data/geonames")
    ap.add_argument("--teacher-cache", default="data/lattice_cache.json")
    ap.add_argument("--out", default="data/lattice_profiles/fine_lattice_profiles.json")
    ap.add_argument("--coverage-out", default="results/lattice_profile_coverage.json")
    ap.add_argument("--report-out", default="results/lattice_profile_population.json")
    ap.add_argument("--require-exhaustive", action="store_true")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    rows = collect_rows(raw_dir, Path(args.geo_dir), Path(args.teacher_cache))
    artifact = merge_rows(rows)
    errors = validate_profile_artifact(artifact)
    if errors:
        raise SystemExit("invalid lattice profile artifact:\n" + "\n".join(errors[:50]))

    write_json(Path(args.out), artifact)
    coverage = coverage_report(artifact)
    if args.coverage_out:
        write_json(Path(args.coverage_out), coverage)
    report = population_report(raw_dir, artifact, Path(args.geo_dir), Path(args.teacher_cache))
    if args.report_out:
        write_json(Path(args.report_out), report)

    profile_total = sum(report["profile_counts"].values())
    print(f"rows={len(rows)} profiles={profile_total} -> {args.out}", flush=True)
    print(
        "population: "
        f"available={len(report['available_sources'])} "
        f"missing={len(report['missing_sources'])} "
        f"unimplemented={len(report['unimplemented_sources'])} "
        f"exhaustive={report['exhaustive']}",
        flush=True,
    )
    if args.require_exhaustive and not report["exhaustive"]:
        print("not exhaustive: missing local source files or parser support", file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
