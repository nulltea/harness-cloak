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
    terms = _terms(path)
    by_id = {term.get("id"): term for term in terms if term.get("id")}

    def family_levels(term_id: str, seen: set[str] | None = None) -> list[str]:
        seen = seen or set()
        if term_id in seen:
            return []
        seen.add(term_id)
        term = by_id.get(term_id, {})
        levels = []
        for parent in term.get("parents", []):
            if parent in family_roots and family_roots[parent] not in levels:
                levels.append(family_roots[parent])
            for level in family_levels(parent, seen):
                if level not in levels:
                    levels.append(level)
        return levels

    for term in terms:
        name = norm(term.get("name", ""))
        if not name:
            continue
        levels = family_levels(term.get("id", ""))
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
