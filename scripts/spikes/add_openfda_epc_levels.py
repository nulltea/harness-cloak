"""Patch drug-health-procedure.proposed.cleaned.json with real openFDA pharm_class EPC levels.

Problem: every drug entry in this file jumps straight from the specific drug name to the
generic "medication" tier (e.g. bupropion -> medication -> pharmaceutical compound -> chemical
substance), because none of the local openFDA-derived builders ever read the `pharm_class` field
(see scripts/lattice_sources/drugs.py:40, which hardcodes levels=["medication"]). The raw NDC
dump has real, FDA-authoritative Established Pharmacologic Class (EPC) tags for 53.8% of records
(e.g. bupropion -> "Aminoketone [EPC]"). This patches ONLY the proposed/cleaned artifact (not the
builder or fine_lattice_profiles.json, per explicit scope decision) by inserting that EPC class
as a new, narrower level with a REAL, source-backed count (distinct generic drug names sharing
the same EPC tag across the whole raw dataset) -- not another fabricated estimate.
"""
from __future__ import annotations

import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

RAW_ZIP = Path("data/lattice_sources/raw/drug/openfda_ndc.json.zip")
TARGET = Path("data/lattice_profiles/proposed/drug-health-procedure.proposed.cleaned.json")
REPORT = Path("data/lattice_profiles/proposed/drug-health-procedure.epc_patch_report.json")

SALT_SUFFIXES = [
    " hydrochloride", " hcl", " hydrobromide", " sulfate", " sodium", " tartrate",
    " succinate", " citrate", " maleate", " mesylate", " phosphate", " acetate",
    " bromide", " besylate", " fumarate", " dihydrochloride",
]


def _norm(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().lower())


def _base_name(name: str) -> str:
    n = _norm(name)
    for suffix in SALT_SUFFIXES:
        if n.endswith(suffix):
            return n[: -len(suffix)].strip()
    return n


def load_epc_index() -> tuple[dict[str, str], dict[str, int]]:
    """base drug name -> most common EPC tag; EPC tag -> count of distinct generic names sharing it."""
    with zipfile.ZipFile(RAW_ZIP) as zf:
        (name,) = zf.namelist()
        with zf.open(name) as f:
            data = json.load(f)

    epc_votes_by_base: dict[str, Counter[str]] = defaultdict(Counter)
    generics_by_epc: dict[str, set[str]] = defaultdict(set)
    for record in data["results"]:
        generic = record.get("generic_name")
        if not generic:
            continue
        # combo products (e.g. "Nite-Time COLD/FLU Medicine") list generic_name as just one
        # ingredient (e.g. "acetaminophen") but pharm_class for the WHOLE combination -- so a
        # plain analgesic like acetaminophen ends up "voting" for an unrelated companion
        # ingredient's class (antihistamine, decongestant, ...). Only single-ingredient records
        # unambiguously attribute pharm_class to the named ingredient.
        if len(record.get("active_ingredients") or []) != 1:
            continue
        epc_tags = [c for c in (record.get("pharm_class") or []) if c.endswith("[EPC]")]
        if not epc_tags:
            continue
        base = _base_name(generic)
        for tag in epc_tags:
            epc_votes_by_base[base][tag] += 1
            generics_by_epc[tag].add(_norm(generic))

    epc_for_base = {base: votes.most_common(1)[0][0] for base, votes in epc_votes_by_base.items()}
    epc_member_count = {tag: len(generics) for tag, generics in generics_by_epc.items()}
    return epc_for_base, epc_member_count


def clean_epc_label(tag: str) -> str:
    return tag.removesuffix(" [EPC]").strip().lower()


def main() -> None:
    epc_for_base, epc_member_count = load_epc_index()
    print(f"loaded EPC index: {len(epc_for_base)} base drug names, {len(epc_member_count)} distinct EPC classes")

    data = json.loads(TARGET.read_text())
    drug_profiles = data["profiles"]["drug"]

    matched = 0
    skipped_already_specific = 0
    skipped_no_match = 0
    anomalies = []
    match_log = []

    for key, row in drug_profiles.items():
        candidates = [key, *row.get("aliases", [])]
        hit = None
        for candidate in candidates:
            base = _base_name(candidate)
            if base in epc_for_base:
                hit = epc_for_base[base]
                break
        if hit is None:
            skipped_no_match += 1
            continue

        epc_label = clean_epc_label(hit)
        levels = row["levels"]
        if epc_label in levels:
            skipped_already_specific += 1
            continue
        # a handful of FDA EPC tags are literally the molecule's own name (e.g. "Progesterone
        # [EPC]" for progesterone) -- inserting that as a "generalization" would just leak the
        # entry's own surface back into its levels.
        if _norm(key) and _norm(key) in _norm(epc_label):
            skipped_already_specific += 1
            continue

        member_count = float(epc_member_count[hit])
        narrowest_existing_count = row["level_counts"][levels[0]] if levels else float("inf")
        if member_count > narrowest_existing_count:
            anomalies.append({
                "entry": key, "epc_label": epc_label, "epc_count": member_count,
                "existing_narrowest": levels[0] if levels else None,
                "existing_narrowest_count": narrowest_existing_count,
            })
            continue  # don't silently misrepresent a broader-than-claimed class; leave entry untouched

        row["levels"] = [epc_label, *levels]
        row["level_counts"][epc_label] = member_count
        row["level_grounding"][epc_label] = {
            "status": "certifying",
            "source_family": "openfda-pharm-class",
            "selector": f"openfda_ndc.pharm_class == '{hit}'",
            "member_set_ref": f"openfda-ndc:pharm_class:{hit}",
            "count_evidence": (
                f"count of distinct FDA generic drug names carrying the pharm_class tag '{hit}' "
                f"in data/lattice_sources/raw/drug/openfda_ndc.json.zip -- real, source-backed count"
            ),
        }
        row["count"] = member_count
        matched += 1
        match_log.append({"entry": key, "epc_label": epc_label, "count": member_count})

    TARGET.write_text(json.dumps(data, indent=2, sort_keys=True))
    report = {
        "matched": matched,
        "skipped_already_specific": skipped_already_specific,
        "skipped_no_epc_match": skipped_no_match,
        "anomalies_epc_count_exceeds_narrower_level": anomalies,
        "sample_matches": match_log[:30],
    }
    REPORT.write_text(json.dumps(report, indent=2))
    print(f"matched {matched} entries with a real EPC level; {skipped_no_match} had no EPC match; "
          f"{skipped_already_specific} already had this exact level; {len(anomalies)} anomalies (see report)")


def _selfcheck() -> None:
    data = json.loads(TARGET.read_text())
    for key, row in data["profiles"]["drug"].items():
        levels = row["levels"]
        counts = [row["level_counts"][lvl] for lvl in levels]
        assert counts == sorted(counts), f"{key} not monotone after EPC patch: {counts}"
        assert len(levels) == len(set(levels)), f"{key} has duplicate levels: {levels}"
    print("selfcheck OK: every drug entry still monotone and deduped after EPC patch")


if __name__ == "__main__":
    main()
    _selfcheck()
