import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from lattice_sources.common import ProfileRow, norm

PCS_SECTION_LEVELS = {
    "0": "medical and surgical procedure",
    "1": "obstetric procedure",
    "2": "placement procedure",
    "3": "administration procedure",
    "4": "measurement and monitoring procedure",
    "5": "extracorporeal assistance procedure",
    "6": "extracorporeal therapy procedure",
    "7": "osteopathic procedure",
    "8": "other medical procedure",
    "9": "chiropractic procedure",
    "B": "imaging procedure",
    "C": "nuclear medicine procedure",
    "D": "radiation therapy procedure",
    "F": "rehabilitation procedure",
    "G": "mental health procedure",
    "H": "substance abuse treatment procedure",
}


def rows_from_icd10_pcs_order_zip(path: Path) -> list[ProfileRow]:
    rows_by_surface = {}
    counts = Counter()
    sources = defaultdict(set)
    with zipfile.ZipFile(path) as zf:
        for name in zf.namelist():
            if not name.lower().endswith(".txt"):
                continue
            for raw in zf.read(name).decode("utf-8", "ignore").splitlines():
                parsed = _parse_order_line(raw)
                if not parsed:
                    continue
                code, surface = parsed
                if not _surface_allowed(surface):
                    continue
                levels = _levels_for_code(code)
                counts[surface] += 1
                sources[surface].add(f"icd10pcs:{code}")
                rows_by_surface[surface] = levels
    return [
        ProfileRow(
            runtime_type="medical-procedure",
            surface=surface,
            aliases=[],
            levels=levels,
            source_ids=sorted(sources[surface]),
            count=max(float(counts[surface]), 1.0),
        )
        for surface, levels in sorted(rows_by_surface.items())
    ]


def _parse_order_line(raw: str) -> tuple[str, str] | None:
    parts = raw.strip().split(maxsplit=3)
    if len(parts) < 4:
        return None
    _order, code, valid, desc = parts
    if valid != "1" or not re.fullmatch(r"[0-9A-HJ-NP-Z]{7}", code):
        return None
    desc_parts = re.split(r"\s{2,}", desc.strip())
    if desc_parts:
        desc = desc_parts[-1]
    surface = norm(desc.replace(",", " "))
    return code, surface or ""


def _levels_for_code(code: str) -> list[str]:
    levels = []
    section = PCS_SECTION_LEVELS.get(code[0].upper())
    if section:
        levels.append(section)
    if "medical procedure" not in levels:
        levels.append("medical procedure")
    return levels


def _surface_allowed(surface: str) -> bool:
    if not surface or any(ch.isdigit() for ch in surface):
        return False
    return len(surface.split()) <= 8
