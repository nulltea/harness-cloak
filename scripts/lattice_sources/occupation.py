import csv
from pathlib import Path

from lattice_sources.common import ProfileRow, norm


def _profession_levels(title: str, major_group: str = "") -> list[str]:
    t = norm(f"{title} {major_group}")
    if "professional" in norm(major_group):
        return ["professional worker"]
    if any(w in t for w in ("journalist", "reporter", "news analyst", "correspondent")):
        return ["media worker"]
    if any(w in t for w in ("medical", "physician", "doctor", "nurse", "health")):
        return ["healthcare worker"]
    if any(w in t for w in ("law", "legal", "judge", "prosecutor")):
        return ["legal professional"]
    if any(w in t for w in ("teacher", "education", "professor", "school")):
        return ["education worker"]
    if "professional" in t or major_group:
        return ["professional worker"]
    return ["worker"]


def rows_from_onet_titles(path: Path) -> list[ProfileRow]:
    rows = []
    with Path(path).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            title = norm(r.get("Title", ""))
            alt = norm(r.get("Alternate Title", ""))
            if not title or not alt:
                continue
            rows.append(ProfileRow(
                runtime_type="profession",
                surface=title,
                aliases=[alt],
                levels=_profession_levels(f"{title} {alt}"),
                source_ids=[f"onet:{r.get('O*NET-SOC Code', '').strip()}"],
            ))
    return rows


def rows_from_isco_csv(path: Path) -> list[ProfileRow]:
    rows = []
    with Path(path).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            title = norm(r.get("title", ""))
            if not title:
                continue
            rows.append(ProfileRow(
                runtime_type="profession",
                surface=title,
                aliases=[],
                levels=_profession_levels(title, r.get("major_group", "")),
                source_ids=[f"isco:{r.get('code', '').strip()}"],
            ))
    return rows
