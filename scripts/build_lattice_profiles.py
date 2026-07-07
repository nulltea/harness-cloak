"""Build fine runtime lattice profiles from cached raw datasets.

Network access is deliberately absent here. Use scripts/fetch_lattice_sources.py or manual downloads
to populate data/lattice_sources/raw first.
"""
import argparse
import json
from collections import defaultdict
from datetime import date
from pathlib import Path

from cloak.lattice_profiles import validate_profile_artifact
from cloak.runtime_types import PLACEHOLDER_ONLY_TYPES
from lattice_sources.categorical import alias_rows
from lattice_sources.common import ProfileRow, norm
from lattice_sources.obo import rows_from_obo
from lattice_sources.occupation import rows_from_isco_csv, rows_from_onet_titles

HEALTH_FAMILY_ROOTS = {
    "DOID:28": "endocrine condition",
    "MONDO:0005151": "infectious disease",
    "MONDO:0005084": "respiratory condition",
    "MONDO:0002025": "mental health condition",
}

CATEGORICAL_ALIASES = {
    "gender": {"female": ["woman"], "male": ["man"]},
    "marital-status": {"married": ["wedded"], "divorced": ["formerly married"], "widowed": ["widow", "widower"]},
    "sexual-orientation": {"gay": ["homosexual"], "bisexual": ["bi"], "lesbian": []},
}


def _add_row(dst: dict, row: ProfileRow) -> None:
    rt = row.runtime_type
    surface = norm(row.surface)
    cur = dst[rt].setdefault(surface, {"aliases": set(), "levels": [], "source_ids": set(), "count": 1.0})
    cur["aliases"].update(a for a in row.aliases if norm(a) and norm(a) != surface)
    for level in row.levels:
        level = norm(level)
        if level and level not in cur["levels"]:
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
    return {"schema_version": 1, "created": str(date.today()), "sources": {}, "profiles": profiles}


def collect_rows(raw_dir: Path) -> list[ProfileRow]:
    rows = []
    onet = raw_dir / "onet" / "Alternate Titles.txt"
    if onet.exists():
        rows.extend(rows_from_onet_titles(onet))
    isco = raw_dir / "isco" / "isco.csv"
    if isco.exists():
        rows.extend(rows_from_isco_csv(isco))
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
    ap.add_argument("--out", default="data/lattice_profiles/fine_lattice_profiles.json")
    ap.add_argument("--coverage-out", default=None)
    args = ap.parse_args()
    rows = collect_rows(Path(args.raw_dir))
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
