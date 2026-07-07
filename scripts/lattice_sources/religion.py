import re
from pathlib import Path

from lattice_sources.common import ProfileRow, norm

RELIGION_LEVEL_BY_CODE = [
    (1000, 1999, ["christian religion", "religious tradition"]),
    (2000, 2999, ["abrahamic religion", "religious tradition"]),
    (3000, 3999, ["islamic religion", "religious tradition"]),
    (4000, 4999, ["religious tradition"]),
    (5000, 5999, ["dharmic religion", "religious tradition"]),
    (6000, 6999, ["dharmic religion", "religious tradition"]),
    (7000, 7999, ["east asian religion", "religious tradition"]),
    (8000, 8999, ["indigenous religion", "religious tradition"]),
    (9000, 9799, ["nonreligion"]),
    (9800, 9899, ["religious tradition"]),
]


def rows_from_arda_stata(path: Path) -> list[ProfileRow]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("ARDA Stata parsing requires pandas in the artifact-build environment") from exc
    reader = pd.read_stata(path, iterator=True)
    return rows_from_arda_variable_labels(reader.variable_labels())


def rows_from_arda_variable_labels(labels: dict[str, str]) -> list[ProfileRow]:
    seen = {}
    for var, label in labels.items():
        parsed = _parse_religion_label(var, label)
        if parsed is None:
            continue
        code, surface = parsed
        row = seen.setdefault(surface, ProfileRow(
            runtime_type="religion",
            surface=surface,
            aliases=[],
            levels=_religion_levels(code),
            source_ids=[f"arda:{code}"],
        ))
        if f"arda:{code}" not in row.source_ids:
            row.source_ids.append(f"arda:{code}")
    return sorted(seen.values(), key=lambda r: r.surface)


def _parse_religion_label(var: str, label: str) -> tuple[int, str] | None:
    if not var.endswith("pp"):
        return None
    match = re.match(r"^(\d{4})-Population of (.+)$", label)
    if not match:
        return None
    code = int(match.group(1))
    surface = norm(match.group(2))
    if surface.startswith("unspecified "):
        return None
    return code, surface


def _religion_levels(code: int) -> list[str]:
    for lo, hi, levels in RELIGION_LEVEL_BY_CODE:
        if lo <= code <= hi:
            return levels
    return ["religious tradition"]
