import csv
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from lattice_sources.common import ProfileRow, norm

TAXONOMY_LEVEL_PREFIXES = {
    "282": "hospital",
    "283": "psychiatric hospital",
    "261": "clinic",
    "291": "medical laboratory",
    "332": "medical equipment supplier",
    "333": "pharmacy",
    "335": "medical supply organization",
    "341": "ambulance service",
    "251": "home health organization",
    "253": "transportation service organization",
    "363": "advanced practice provider organization",
}


def rows_from_nppes_zip(path: Path) -> list[ProfileRow]:
    rows = []
    counts = Counter()
    sources = defaultdict(set)
    records = []
    with zipfile.ZipFile(path) as zf:
        for name in zf.namelist():
            if not name.lower().endswith(".csv"):
                continue
            with zf.open(name) as f:
                text = (line.decode("utf-8", "ignore") for line in f)
                reader = csv.DictReader(text)
                for record in reader:
                    if str(record.get("Entity Type Code", "")).strip() != "2":
                        continue
                    surface = norm(record.get("Provider Organization Name (Legal Business Name)", ""))
                    if not _surface_allowed(surface):
                        continue
                    aliases = sorted({
                        alias
                        for alias in [norm(record.get("Provider Other Organization Name", ""))]
                        if alias and alias != surface and _surface_allowed(alias)
                    })
                    levels = _levels_for_record(record)
                    npi = str(record.get("NPI", "")).strip()
                    records.append((surface, aliases, levels, npi))
                    counts[surface] += 1
                    if npi:
                        sources[surface].add(f"nppes:{npi}")
    for surface, aliases, levels, npi in records:
        rows.append(ProfileRow(
            runtime_type="organization-medical-facility",
            surface=surface,
            aliases=aliases,
            levels=levels,
            source_ids=sorted(sources[surface] or ({f"nppes:{npi}"} if npi else set())),
            count=max(float(counts[surface]), 1.0),
        ))
    return rows


def _levels_for_record(record: dict) -> list[str]:
    levels = []
    for key, value in record.items():
        if not key.startswith("Healthcare Provider Taxonomy Code"):
            continue
        code = str(value or "").strip()
        for prefix, level in TAXONOMY_LEVEL_PREFIXES.items():
            if code.startswith(prefix) and level not in levels:
                levels.append(level)
    if "healthcare organization" not in levels:
        levels.append("healthcare organization")
    return levels


def _surface_allowed(surface: str) -> bool:
    if not surface or len(surface) > 80:
        return False
    if surface in {"<unavail>", "unavail", "n/a", "na"}:
        return False
    if "," in surface or any(ch.isdigit() for ch in surface):
        return False
    if len(surface.split()) > 8:
        return False
    return True
