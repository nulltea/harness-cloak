"""Deterministic reference-source lookups for lattice level generation.

Per the coherence-hardening plan's non-negotiable constraint: a real local dataset must be tried
before any model call. A hit here always carries a real `member_set` (so `compile_level_counts`
marks it `status: certifying`, not a guess to grade later) and, for health-condition and
medical-procedure, the level *text* itself, not just a count to attach to a model-proposed label.

Three loaders, one per runtime type that has a real local source today:
- drug: FDA openFDA NDC Directory `pharm_class` (Established Pharmacologic Class).
- health-condition: Human Disease Ontology (DOID) `is_a` hierarchy.
- medical-procedure: CMS ICD-10-PCS order file, position-encoded code hierarchy.

Each loader is read once and cached (`lru_cache`) -- these are large files (12MB ICD-10-PCS
text, 167k-line OBO file, 244MB NDC zip) and every queue item for a runtime type shares one
parsed index.
"""
from __future__ import annotations

import json
import re
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

DEFAULT_OPENFDA_NDC_ZIP = Path("data/lattice_sources/raw/drug/openfda_ndc.json.zip")
DEFAULT_DOID_OBO = Path("data/lattice_sources/raw/health/doid.obo")
DEFAULT_ICD10PCS_ZIP = Path("data/lattice_sources/raw/procedure/icd10pcs_order_2026.zip")

SALT_SUFFIXES = [
    " hydrochloride", " hcl", " hydrobromide", " sulfate", " sodium", " tartrate",
    " succinate", " citrate", " maleate", " mesylate", " phosphate", " acetate",
    " bromide", " besylate", " fumarate", " dihydrochloride",
]


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def _base_drug_name(name: str) -> str:
    n = _norm(name)
    for suffix in SALT_SUFFIXES:
        if n.endswith(suffix):
            return n[: -len(suffix)].strip()
    return n


# ---------------------------------------------------------------------------
# drug: openFDA NDC pharm_class (EPC)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OpenFdaEpcEntry:
    tag: str  # e.g. "Aminoketone [EPC]"
    members: frozenset[str]  # normalized generic drug names sharing this tag


@lru_cache(maxsize=4)
def load_openfda_pharm_class_index(raw_zip_path: str = str(DEFAULT_OPENFDA_NDC_ZIP)) -> dict[str, OpenFdaEpcEntry]:
    """base drug name (normalized, salt-stripped) -> its most common real EPC tag + member set.

    Only single-active-ingredient records are indexed: combo OTC products (e.g. "Nite-Time
    COLD/FLU Medicine") list `generic_name` as just one ingredient (e.g. "acetaminophen") but
    `pharm_class` for the WHOLE combination, so a plain analgesic would otherwise "vote" for an
    unrelated companion ingredient's class (this reproduced for real: acetaminophen -> antihistamine).
    """
    path = Path(raw_zip_path)
    if not path.exists():
        return {}
    with zipfile.ZipFile(path) as zf:
        (name,) = zf.namelist()
        with zf.open(name) as f:
            data = json.load(f)

    votes_by_base: dict[str, Counter[str]] = defaultdict(Counter)
    members_by_tag: dict[str, set[str]] = defaultdict(set)
    for record in data.get("results", []):
        generic = record.get("generic_name")
        if not generic or len(record.get("active_ingredients") or []) != 1:
            continue
        epc_tags = [c for c in (record.get("pharm_class") or []) if c.endswith("[EPC]")]
        if not epc_tags:
            continue
        base = _base_drug_name(generic)
        for tag in epc_tags:
            votes_by_base[base][tag] += 1
            members_by_tag[tag].add(_norm(generic))

    index: dict[str, OpenFdaEpcEntry] = {}
    for base, votes in votes_by_base.items():
        tag, _ = votes.most_common(1)[0]
        index[base] = OpenFdaEpcEntry(tag=tag, members=frozenset(members_by_tag[tag]))
    return index


def epc_label(tag: str) -> str:
    return tag.removesuffix(" [EPC]").strip().lower()


def lookup_openfda_reference(item: dict[str, Any], *, raw_zip_path: str = str(DEFAULT_OPENFDA_NDC_ZIP)) -> list[dict[str, Any]] | None:
    index = load_openfda_pharm_class_index(raw_zip_path)
    if not index:
        return None
    surface = str(item.get("surface") or item.get("canonical_value") or "")
    candidates = [surface, *item.get("aliases", [])]
    hit_base = None
    for candidate in candidates:
        base = _base_drug_name(str(candidate))
        if base in index:
            hit_base = base
            break
    if hit_base is None:
        return None

    entry = index[hit_base]
    label = epc_label(entry.tag)
    # a handful of FDA EPC tags are literally the molecule's own name (e.g. "Progesterone
    # [EPC]" for progesterone) -- that's not a generalization, it's a self-leak.
    surface_norm = _norm(surface)
    if surface_norm and surface_norm in _norm(label):
        return None

    return [
        {
            "level": label,
            "source_family": "openfda-pharm-class",
            "selector": f"openfda_ndc.pharm_class == '{entry.tag}'",
            "member_set": entry.members,
            "member_set_ref": f"openfda-ndc:pharm_class:{entry.tag}",
        }
    ]


# ---------------------------------------------------------------------------
# health-condition: Disease Ontology (DOID) is_a hierarchy
# ---------------------------------------------------------------------------


@dataclass
class DoidNode:
    id: str
    name: str
    parents: list[str] = field(default_factory=list)


_TERM_ID_RE = re.compile(r"^id:\s*(DOID:\d+)", re.M)
_TERM_NAME_RE = re.compile(r"^name:\s*(.+)$", re.M)
_TERM_ISA_RE = re.compile(r"^is_a:\s*(DOID:\d+)", re.M)

DOID_ROOT = "DOID:4"  # "disease" -- the ontology root, treated as a ceiling, not a mid-chain rung


@lru_cache(maxsize=4)
def load_doid_index(obo_path: str = str(DEFAULT_DOID_OBO)) -> dict[str, DoidNode]:
    path = Path(obo_path)
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")
    nodes: dict[str, DoidNode] = {}
    for stanza in text.split("[Term]\n")[1:]:
        id_match = _TERM_ID_RE.search(stanza)
        name_match = _TERM_NAME_RE.search(stanza)
        if not id_match or not name_match:
            continue
        node_id = id_match.group(1)
        nodes[node_id] = DoidNode(
            id=node_id,
            name=name_match.group(1).strip(),
            parents=_TERM_ISA_RE.findall(stanza),
        )
    return nodes


@lru_cache(maxsize=4)
def _doid_children_index(obo_path: str = str(DEFAULT_DOID_OBO)) -> dict[str, list[str]]:
    nodes = load_doid_index(obo_path)
    children: dict[str, list[str]] = defaultdict(list)
    for node in nodes.values():
        for parent in node.parents:
            children[parent].append(node.id)
    return dict(children)


def _doid_descendants(node_id: str, children_index: dict[str, list[str]]) -> set[str]:
    seen = {node_id}
    stack = [node_id]
    while stack:
        current = stack.pop()
        for child in children_index.get(current, []):
            if child not in seen:
                seen.add(child)
                stack.append(child)
    return seen


@lru_cache(maxsize=4)
def _doid_name_index(obo_path: str = str(DEFAULT_DOID_OBO)) -> dict[str, str]:
    return {_norm(node.name): node.id for node in load_doid_index(obo_path).values()}


def lookup_doid_reference(
    item: dict[str, Any], *, obo_path: str = str(DEFAULT_DOID_OBO), max_hops: int = 4
) -> list[dict[str, Any]] | None:
    nodes = load_doid_index(obo_path)
    if not nodes:
        return None
    name_index = _doid_name_index(obo_path)
    children_index = _doid_children_index(obo_path)

    surface = str(item.get("surface") or item.get("canonical_value") or "")
    candidates = [surface, *item.get("aliases", [])]
    node_id = None
    for candidate in candidates:
        node_id = name_index.get(_norm(str(candidate)))
        if node_id:
            break
    if node_id is None:
        return None

    chain: list[dict[str, Any]] = []
    current = node_id
    seen_ids = set()
    for _ in range(max_hops):
        node = nodes.get(current)
        if not node or not node.parents:
            break
        parent_id = node.parents[0]
        if parent_id == DOID_ROOT or parent_id in seen_ids:
            break  # the ontology root is a ceiling, not a useful mid-chain rung
        seen_ids.add(parent_id)
        parent = nodes.get(parent_id)
        if not parent:
            break
        descendants = _doid_descendants(parent_id, children_index)
        chain.append(
            {
                "level": _norm(parent.name),
                "source_family": "doid-is-a",
                "selector": f"doid.is_a_descendants({parent_id})",
                "member_set": frozenset(descendants),
                "member_set_ref": f"doid:is_a_descendants:{parent_id}",
            }
        )
        current = parent_id
    return chain or None


# ---------------------------------------------------------------------------
# medical-procedure: ICD-10-PCS order file, position-encoded code hierarchy
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Icd10PcsRow:
    code: str
    is_header: bool
    short_desc: str
    long_desc: str


def _parse_icd10pcs_rows(zip_path: str) -> list[Icd10PcsRow]:
    """CMS's fixed-width order file layout (confirmed against the real 2026 file): seq number
    at [0:5], 7-char code (space-padded for header rows) at [6:13], header flag at [14], 61-char
    short description at [16:77], long description in the remainder. Flag "0" is a header row
    carrying an abbreviated prefix (Section / Body System / Root Operation summary, e.g. code
    "001" = "Central Nervous System and Cranial Nerves, Bypass"); flag "1" is a fully specified,
    billable 7-character code.
    """
    path = Path(zip_path)
    if not path.exists():
        return []
    with zipfile.ZipFile(path) as zf:
        (member,) = [n for n in zf.namelist() if n.endswith("_order_2026.txt") and "addenda" not in n]
        text = zf.read(member).decode("utf-8", errors="replace")

    rows = []
    for line in text.splitlines():
        if len(line) < 77:
            continue
        code = line[6:13].strip()
        flag = line[14:15]
        if not code or flag not in ("0", "1"):
            continue
        rows.append(
            Icd10PcsRow(code=code, is_header=(flag == "0"), short_desc=line[16:77].strip(), long_desc=line[77:].strip())
        )
    return rows


@lru_cache(maxsize=4)
def load_icd10pcs_index(zip_path: str = str(DEFAULT_ICD10PCS_ZIP)) -> dict[str, tuple[str, frozenset[str]]]:
    """prefix -> (header description, set of full 7-char codes sharing it). Tries prefix lengths
    1, 3, and 4 (Section / Section+BodySystem+RootOperation / +BodyPart), but only emits a
    prefix that has a real header row of its own -- the confirmed 2026 order file only carries
    3-character header rows (914 of them, one per Section+BodySystem+RootOperation combination),
    no 1- or 4-character ones, so in practice only length-3 prefixes are populated. Deliberately
    not backfilling a hand-written Section name table here: every level this loader emits must
    trace to an actual row in the source file, not an assumption about its structure."""
    rows = _parse_icd10pcs_rows(zip_path)
    header_desc: dict[str, str] = {}
    members: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        for prefix_len in (1, 3, 4):
            prefix = row.code[:prefix_len]
            if row.is_header and prefix == row.code:
                header_desc[prefix] = row.long_desc
            if not row.is_header:
                members[prefix].add(row.code)
    return {prefix: (header_desc.get(prefix, prefix), frozenset(codes)) for prefix, codes in members.items() if prefix in header_desc}


@lru_cache(maxsize=4)
def _icd10pcs_description_index(zip_path: str = str(DEFAULT_ICD10PCS_ZIP)) -> dict[str, str]:
    """normalized short/long description (of a full, non-header code) -> that 7-char code.

    Exact match only, deliberately -- fuzzy/substring matching against free-text procedure
    surfaces is exactly the kind of shortcut that produced wrong matches elsewhere this session
    (e.g. "ap" matching an unrelated drug via a noisy alias). A surface that doesn't exactly
    match a real ICD-10-PCS description falls through to the model instead of a risky guess.
    """
    index: dict[str, str] = {}
    for row in _parse_icd10pcs_rows(zip_path):
        if row.is_header:
            continue
        index.setdefault(_norm(row.long_desc), row.code)
        index.setdefault(_norm(row.short_desc), row.code)
    return index


_SOURCE_ID_CODE_RE = re.compile(r"^icd10pcs:([A-Z0-9]{7})$", re.I)


def _code_from_source_ids(item: dict[str, Any]) -> str | None:
    for source_id in item.get("source_ids") or []:
        match = _SOURCE_ID_CODE_RE.match(str(source_id))
        if match:
            return match.group(1).upper()
    return None


def lookup_icd10pcs_reference(item: dict[str, Any], *, zip_path: str = str(DEFAULT_ICD10PCS_ZIP)) -> list[dict[str, Any]] | None:
    index = load_icd10pcs_index(zip_path)
    if not index:
        return None

    code = _code_from_source_ids(item)
    if code is None:
        description_index = _icd10pcs_description_index(zip_path)
        surface = str(item.get("surface") or item.get("canonical_value") or "")
        for candidate in [surface, *item.get("aliases", [])]:
            code = description_index.get(_norm(str(candidate)))
            if code:
                break
    if code is None or len(code) != 7:
        return None

    chain: list[dict[str, Any]] = []
    for prefix_len in (4, 3, 1):
        prefix = code[:prefix_len]
        hit = index.get(prefix)
        if not hit:
            continue
        desc, members = hit
        chain.append(
            {
                "level": _norm(desc),
                "source_family": "icd10pcs-prefix",
                "selector": f"icd10pcs.prefix({prefix})",
                "member_set": members,
                "member_set_ref": f"icd10pcs:prefix:{prefix}",
            }
        )
    return chain or None


# ---------------------------------------------------------------------------
# dispatcher
# ---------------------------------------------------------------------------

_LOOKUPS = {
    "drug": lookup_openfda_reference,
    "health-condition": lookup_doid_reference,
    "medical-procedure": lookup_icd10pcs_reference,
}


def reference_candidates_for(item: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Try the registered real-data loader for this item's runtime type. Returns None (never an
    empty list) when there's no loader or no match, so callers can cleanly fall through to the
    next local source and finally the model."""
    runtime_type = str(item.get("runtime_type") or "")
    lookup = _LOOKUPS.get(runtime_type)
    if lookup is None:
        return None
    return lookup(item)
