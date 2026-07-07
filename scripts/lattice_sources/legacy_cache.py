import json
from pathlib import Path

from lattice_sources.common import ProfileRow, norm

ORG_CUES = (
    "organization",
    "institution",
    "company",
)

WEAK_UNTYPED_ORG_SURFACE_CUES = (
    "unicorn",
    "breeding start-up",
    "breeding startup",
)


def rows_from_legacy_teacher_cache(path: Path) -> list[ProfileRow]:
    path = Path(path)
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    rows = []
    for key, row in sorted(data.items()):
        runtime_type, surface = _typed_key(key)
        levels = [norm(x) for x in row.get("lattice", []) if norm(x)]
        if not levels:
            continue
        if runtime_type is None:
            if _weak_untyped_org_surface(surface):
                continue
            runtime_type = _infer_runtime_type(levels)
        if runtime_type != "ORG":
            continue
        rows.append(ProfileRow(
            runtime_type=runtime_type,
            surface=norm(surface),
            aliases=[],
            levels=levels,
            source_ids=[f"legacy-teacher-cache:{norm(surface)}"],
            count=1.0,
        ))
    return rows


def _typed_key(key: str) -> tuple[str | None, str]:
    if "::" not in key:
        return None, key
    runtime_type, surface = key.split("::", 1)
    return runtime_type, surface


def _infer_runtime_type(levels: list[str]) -> str | None:
    text = " ".join(levels)
    if any(cue in text for cue in ORG_CUES):
        return "ORG"
    return None


def _weak_untyped_org_surface(surface: str) -> bool:
    surface = norm(surface)
    return len(surface) <= 3 or any(cue in surface for cue in WEAK_UNTYPED_ORG_SURFACE_CUES)
