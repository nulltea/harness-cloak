"""Build producer-ready lattice repairs from environment and QA audit evidence."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

from cloak.lattice.producer.queue import normalize_item
from cloak.lattice.producer.reference_sources import reference_candidates_for
from cloak.lattice.core import NLI_THRESH


def _norm(value: object) -> str:
    return " ".join(str(value or "").lower().split())


def _events(audits: Iterable[Mapping[str, object]]) -> list[Mapping[str, object]]:
    return [event for audit in audits for event in audit.get("events", [])
            if isinstance(event, Mapping)]


def _profile_entry(profiles: Mapping[str, object], runtime_type: str, surface: str) -> tuple[str, Mapping[str, object]] | None:
    entries = profiles.get("profiles", {}).get(runtime_type, {})
    if not isinstance(entries, Mapping):
        return None
    wanted = _norm(surface)
    for canonical, row in entries.items():
        if not isinstance(row, Mapping):
            continue
        if _norm(canonical) == wanted or any(_norm(alias) == wanted for alias in row.get("aliases", [])):
            return str(canonical), row
    return None


def _marked_excerpt(excerpt: object, surface: str) -> str | None:
    if not isinstance(excerpt, str) or not excerpt.strip() or not surface:
        return None
    marked, count = re.subn(re.escape(surface), f"[SPAN]{surface}[/SPAN]", excerpt, count=1, flags=re.I)
    return marked if count else None


def _self_type_entailed(evidence: Mapping[str, object]) -> bool:
    """Accept only the shared NLI operating point, not merely a formable job."""
    try:
        score = float(evidence.get("self_type_score"))
    except (TypeError, ValueError):
        return False
    return not bool(evidence.get("prep_filtered")) and math.isfinite(score) and score >= NLI_THRESH


# Prompt-ready slice of each audit code's evidence. Only fields the producer can act on when
# rebuilding a ladder; everything else stays in repair_evidence for human review.
_FINDING_FIELDS: dict[str, tuple[str, ...]] = {
    "lattice_level_suspect": ("unreadable_level", "readable_coarser_level", "chain", "scores"),
    "controlled_ladder_issue": ("issues", "levels", "level_counts", "profile_count",
                                "first_level_log10_jump", "max_adjacent_log10_jump"),
    "coarse_or_degenerate_action_menu": ("levels",),
}

_MAX_REPAIR_FINDINGS = 5


def distill_findings(evidence_records: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    """Bounded, deduplicated extraction of what each ladder audit actually observed."""
    findings: list[dict[str, object]] = []
    seen: set[str] = set()
    for record in evidence_records:
        fields = _FINDING_FIELDS.get(str(record.get("code", "")))
        evidence = record.get("evidence")
        if fields is None or not isinstance(evidence, Mapping):
            continue
        finding: dict[str, object] = {"code": str(record["code"])}
        finding.update({field: evidence[field] for field in fields if evidence.get(field) is not None})
        key = json.dumps(finding, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        findings.append(finding)
        if len(findings) >= _MAX_REPAIR_FINDINGS:
            break
    return findings


def _reprocess_guidance(target: Mapping[str, object]) -> tuple[str, str]:
    """Return a stable producer reason and bounded, evidence-derived guidance."""
    runtime_type = str(target["runtime_type"])
    if target["repair_kind"] == "missing_profile":
        if target.get("source_backed"):
            return (
                "missing_profile_reference_backed",
                f"A trusted local source resolves this {runtime_type} surface, but no certified "
                "lattice profile exists. Build the profile from the reference-backed identity and "
                "preserve only truthful, source-supported hierarchy and count evidence.",
            )
        return (
            "missing_profile_type_entailed",
            f"The marked context supports {runtime_type} membership, but no certified lattice "
            "profile exists. Create or extend a profile only when the surface has a stable "
            "referent; otherwise mark the surface low-confidence or ambiguous.",
        )

    reason_codes = ", ".join(str(code) for code in target["repair_reason_codes"])
    return (
        "lattice_structure_or_utility_failure",
        "Re-evaluate this existing profile's truthful hierarchy and anonymity counts. "
        f"Audit evidence: {reason_codes}. Repair the diagnosed failure without flattening the "
        "ladder or inventing levels unsupported by the entry and source evidence.",
    )


def _triage_guidance(runtime_type: str) -> tuple[str, str]:
    return (
        "unprofiled_type_entailed_unresolved",
        f"The marked context supports {runtime_type} membership, but no trusted local source "
        "resolved a stable profile identity. Do not create a profile automatically; review or "
        "promote this item only after obtaining canonical identity evidence.",
    )


def build_repair_queue(
    environment_audits: Iterable[Mapping[str, object]],
    qa_audits: Iterable[Mapping[str, object]],
    *,
    profiles_path: str | Path,
    reference_lookup=reference_candidates_for,
) -> dict[str, list[dict[str, object]]]:
    """Return producer, triage, manual-review, and non-producer audit records.

    Only a reference-backed unprofiled span or a profile-specific ladder defect enters the
    producer queue. Type-entailing but identity-unresolved spans are preserved in a non-runnable
    triage queue. Detector uncertainty, profile conflicts, and teacher/reader issues are never
    reinterpreted as a request to fabricate a lattice entry.
    """
    profiles = json.loads(Path(profiles_path).read_text())
    environment_events = _events(environment_audits)
    qa_events = _events(qa_audits)
    self_type: dict[tuple[str, str, str], Mapping[str, object]] = {}
    for event in environment_events:
        if event.get("code") != "unprofiled_self_type_diagnostic":
            continue
        refs = event.get("entity_refs") or {}
        if isinstance(refs, Mapping):
            self_type[(str(event.get("doc_id")), str(refs.get("runtime_type")), _norm(refs.get("surface")))] = event

    targets: dict[tuple[str, str], dict[str, object]] = {}
    triage: dict[tuple[str, str], dict[str, object]] = {}
    manual_review: list[dict[str, object]] = []
    non_producer: list[dict[str, object]] = []

    def add_target(*, runtime_type: str, surface: str, entry: str | None, kind: str,
                   code: str, event: Mapping[str, object], excerpt: object = None) -> None:
        profile = _profile_entry(profiles, runtime_type, entry or surface)
        canonical, row = profile if profile is not None else (entry or surface, {})
        key = (runtime_type, _norm(canonical))
        target = targets.setdefault(key, {
            "runtime_type": runtime_type,
            "surface": canonical,
            "canonical_value": canonical,
            "aliases": list(row.get("aliases", [])) if isinstance(row, Mapping) else [],
            "repair_kind": kind,
            "repair_reason_codes": [],
            "repair_evidence": [],
        })
        if kind == "ladder_repair":
            target["repair_kind"] = kind
        target["repair_reason_codes"].append(code)
        target["repair_evidence"].append({
            "doc_id": event.get("doc_id"), "code": code,
            "evidence": event.get("evidence", {}),
        })
        marked = _marked_excerpt(excerpt, surface)
        if marked and not target.get("marked_context_sentence"):
            target["marked_context_sentence"] = marked

    def add_triage(*, runtime_type: str, surface: str, event: Mapping[str, object],
                   excerpt: object, reference_status: str) -> None:
        key = (runtime_type, _norm(surface))
        reason, hint = _triage_guidance(runtime_type)
        item = triage.setdefault(key, {
            "runtime_type": runtime_type,
            "surface": surface,
            "canonical_value": surface,
            "task_kind": "level-proposal",
            "eligible": False,
            "skip_reason": "requires_profile_identity_evidence",
            "reprocess_reason": reason,
            "reprocess_hint": hint,
            "reference_lookup": reference_status,
            "triage_evidence": [],
        })
        item["triage_evidence"].append({
            "doc_id": event.get("doc_id"),
            "code": event.get("code"),
            "evidence": event.get("evidence", {}),
        })
        marked = _marked_excerpt(excerpt, surface)
        if marked and not item.get("marked_context_sentence"):
            item["marked_context_sentence"] = marked

    for event in environment_events:
        code = str(event.get("code", ""))
        refs = event.get("entity_refs") or {}
        if not isinstance(refs, Mapping):
            continue
        runtime_type, surface = str(refs.get("runtime_type", "")), str(refs.get("surface", ""))
        if code == "unprofiled_profile_backed_span" and runtime_type and surface:
            diagnostic = self_type.get((str(event.get("doc_id")), runtime_type, _norm(surface)))
            if diagnostic is None:
                manual_review.append({"doc_id": event.get("doc_id"), "runtime_type": runtime_type,
                                      "surface": surface, "reason_codes": ["unprofiled_self_type_missing"]})
            else:
                evidence = diagnostic.get("evidence") or {}
                if isinstance(evidence, Mapping) and _self_type_entailed(evidence):
                    try:
                        source_hit = bool(reference_lookup({
                            "runtime_type": runtime_type,
                            "surface": surface,
                            "canonical_value": surface,
                        }))
                    except Exception:
                        source_hit = False
                    if source_hit:
                        add_target(runtime_type=runtime_type, surface=surface, entry=None,
                                   kind="missing_profile", code=code, event=event,
                                   excerpt=evidence.get("source_excerpt"))
                        targets[(runtime_type, _norm(surface))]["source_backed"] = True
                    else:
                        add_triage(runtime_type=runtime_type, surface=surface, event=event,
                                   excerpt=evidence.get("source_excerpt"), reference_status="miss")
                else:
                    manual_review.append({"doc_id": event.get("doc_id"), "runtime_type": runtime_type,
                                          "surface": surface,
                                          "reason_codes": ["unprofiled_self_type_unentailed"]})
        elif code in {"controlled_ladder_issue", "coarse_or_degenerate_action_menu"}:
            entry = refs.get("entry") or refs.get("surface")
            if runtime_type and entry:
                add_target(runtime_type=runtime_type, surface=surface or str(entry), entry=str(entry),
                           kind="ladder_repair", code=code, event=event)
        elif code in {"controlled_subspan_profile_conflict", "cross_profile_coreference_candidate"}:
            manual_review.append({"doc_id": event.get("doc_id"), "runtime_type": runtime_type,
                                  "surface": surface, "reason_codes": [code], "evidence": event.get("evidence", {})})
        elif code in {"post_detection_rejection", "detector_to_walk_drop", "controlled_self_type_diagnostic"}:
            non_producer.append({"doc_id": event.get("doc_id"), "code": code,
                                 "entity_refs": dict(refs), "evidence": event.get("evidence", {})})

    for event in qa_events:
        if event.get("code") != "lattice_level_suspect":
            continue
        evidence = event.get("evidence") or {}
        if not isinstance(evidence, Mapping):
            continue
        runtime_type, surface = str(evidence.get("runtime_type", "")), str(evidence.get("surface", ""))
        if runtime_type and surface:
            add_target(runtime_type=runtime_type, surface=surface, entry=None,
                       kind="ladder_repair", code="lattice_level_suspect", event=event)

    producer_items = []
    for target in sorted(targets.values(), key=lambda row: (str(row["runtime_type"]), str(row["canonical_value"]))):
        target["repair_reason_codes"] = sorted(set(target["repair_reason_codes"]))
        findings = distill_findings(target["repair_evidence"])
        if findings:
            target["repair_findings"] = findings
        target["reprocess_reason"], target["reprocess_hint"] = _reprocess_guidance(target)
        target["force_model_proposal"] = True
        target["task_kind"] = "level-proposal"
        target["item_id"] = f"repair:{target['runtime_type']}:{target['canonical_value']}"
        target["repair_priority"] = "high" if target["repair_kind"] == "ladder_repair" else "normal"
        producer_items.append(normalize_item(target, len(producer_items)))
    triage_items = []
    for item in sorted(triage.values(), key=lambda row: (str(row["runtime_type"]), str(row["canonical_value"]))):
        item["item_id"] = f"triage:{item['runtime_type']}:{item['canonical_value']}"
        triage_items.append(normalize_item(item, len(triage_items)))
    manual_review.sort(key=lambda row: (str(row.get("runtime_type", "")), str(row.get("surface", ""))))
    return {
        "producer_items": producer_items,
        "triage_items": triage_items,
        "manual_review": manual_review,
        "non_producer": non_producer,
    }
