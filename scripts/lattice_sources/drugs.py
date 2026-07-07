import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from lattice_sources.common import ProfileRow, norm


def rows_from_openfda_ndc_zip(path: Path) -> list[ProfileRow]:
    records = []
    with zipfile.ZipFile(path) as zf:
        for name in zf.namelist():
            if not name.endswith(".json"):
                continue
            data = json.loads(zf.read(name).decode("utf-8"))
            records.extend(data.get("results", []))

    names_by_product = []
    counts = Counter()
    sources = defaultdict(set)
    for record in records:
        product_id = str(record.get("product_ndc") or record.get("package_ndc") or "").strip()
        names = _record_names(record)
        if not names:
            continue
        names_by_product.append((product_id, names))
        for name in names:
            counts[name] += 1
            if product_id:
                sources[name].add(f"openfda-ndc:{product_id}")

    rows = []
    for product_id, names in names_by_product:
        source_ids = [f"openfda-ndc:{product_id}"] if product_id else []
        for surface in names:
            rows.append(ProfileRow(
                runtime_type="drug",
                surface=surface,
                aliases=sorted(n for n in names if n != surface),
                levels=["medication"],
                source_ids=sorted(sources.get(surface) or source_ids),
                count=max(float(counts[surface]), 1.0),
            ))
    return rows


def _record_names(record: dict) -> list[str]:
    names = [
        record.get("brand_name", ""),
        record.get("generic_name", ""),
    ]
    for ingredient in record.get("active_ingredients", []) or []:
        if isinstance(ingredient, dict):
            names.append(ingredient.get("name", ""))
    names.extend(record.get("substance_name", []) or [])
    out = []
    for name in names:
        name = norm(name)
        if name and _name_allowed(name) and name not in out:
            out.append(name)
    return out


def _name_allowed(name: str) -> bool:
    if any(ch.isdigit() for ch in name):
        return False
    if len(name.split()) > 4:
        return False
    return True
