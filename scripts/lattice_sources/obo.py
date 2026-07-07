import re
from pathlib import Path

from lattice_sources.common import ProfileRow, norm


def _terms(path: Path) -> list[dict]:
    terms = []
    cur = None
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line == "[Term]":
            if cur:
                terms.append(cur)
            cur = {"synonyms": [], "parents": []}
            continue
        if cur is None or not line:
            continue
        if line.startswith("id: "):
            cur["id"] = line[4:].strip()
        elif line.startswith("name: "):
            cur["name"] = line[6:].strip()
        elif line.startswith("synonym: "):
            m = re.search(r'"([^"]+)"', line)
            if m:
                cur["synonyms"].append(m.group(1))
        elif line.startswith("is_a: "):
            cur["parents"].append(line[6:].split()[0])
    if cur:
        terms.append(cur)
    return terms


def rows_from_obo(path: Path, runtime_type: str, family_roots: dict[str, str]) -> list[ProfileRow]:
    rows = []
    for term in _terms(path):
        name = norm(term.get("name", ""))
        if not name:
            continue
        levels = [family_roots[p] for p in term.get("parents", []) if p in family_roots]
        if not levels:
            continue
        rows.append(ProfileRow(
            runtime_type=runtime_type,
            surface=name,
            aliases=[norm(s) for s in term.get("synonyms", []) if norm(s) != name],
            levels=levels,
            source_ids=[term.get("id", "")],
        ))
    return rows
