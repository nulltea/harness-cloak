"""Deterministic audit reports for the QA-v2 environment and utility build.

The reports are diagnostic sidecars.  They never alter detector admission,
relation compilation, reader scoring, or ranker reward aggregation.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from cloak import lattice_profiles as lp
from cloak import span_gate
from cloak.profile_match import DEFAULT_MODEL_ID, SIM_FLOOR, _st_model
from cloak.detect import QA_V2_CLINICAL_LABELS
from cloak.profile_match import PROFILE_BACKED_TYPES
from cloak.runtime_types import PLACEHOLDER_RE

ENVIRONMENT_AUDIT_VERSION = "qa-environment-audit-v2"
QA_AUDIT_VERSION = "qa-build-audit-v1"

_ACTION_FOR_FIX_CLASS = {
    "data_lattice": "rebuild_environment",
    "teacher_redraw": "teacher_repair",
    "reader": "reader_regate",
    "ontology_review": "manual_review",
    "ambiguity": "manual_review",
}


def _norm(value: object) -> str:
    return " ".join(str(value or "").lower().split())


def _stable_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _event(
    *,
    doc_id: str,
    stage: str,
    code: str,
    severity: str,
    fix_class: str,
    entity_refs: Mapping[str, object] | None = None,
    evidence: Mapping[str, object] | None = None,
    recommended_action: str,
) -> dict[str, object]:
    payload = {
        "doc_id": doc_id,
        "stage": stage,
        "code": code,
        "entity_refs": dict(entity_refs or {}),
        "evidence": dict(evidence or {}),
        "recommended_action": recommended_action,
    }
    return {
        "event_id": _stable_hash(payload),
        **payload,
        "severity": severity,
        "fix_class": fix_class,
    }


def _real_levels(row: Mapping[str, object]) -> list[str]:
    lattice = row.get("lattice")
    if not isinstance(lattice, list):
        return []
    return [str(level) for level in lattice if isinstance(level, str)
            and not PLACEHOLDER_RE.fullmatch(level)]


def _detector_runtime_type(row: Mapping[str, object]) -> str:
    """Resolve a detector diagnostic's runtime type across its two schemas."""
    explicit_type = row.get("type") or row.get("runtime_type") or row.get("proposed_runtime_type")
    if explicit_type:
        return str(explicit_type)
    return str(QA_V2_CLINICAL_LABELS.get(str(row.get("raw_label") or ""), ""))


def _detector_row_key(row: Mapping[str, object]) -> tuple[int, int, str]:
    return (
        int(row.get("start", -1)),
        int(row.get("end", -1)),
        _detector_runtime_type(row),
    )


def _whole_word_contains(left: str, right: str) -> bool:
    left_tokens = set(_norm(left).split())
    right_tokens = set(_norm(right).split())
    return bool(left_tokens and right_tokens and (left_tokens <= right_tokens or right_tokens <= left_tokens))


def _finalize(
    version: str,
    events: list[dict[str, object]],
    *,
    observations: list[dict[str, object]] | None = None,
    metadata: Mapping[str, object] | None = None,
) -> dict:
    events = sorted(events, key=lambda row: (
        str(row["doc_id"]), str(row["stage"]), str(row["code"]), str(row["event_id"]),
    ))
    observations = sorted(observations or [], key=lambda row: (
        str(row["doc_id"]), str(row["stage"]), str(row["code"]), str(row["observation_id"]),
    ))
    summary_by_code = dict(sorted(Counter(str(row["code"]) for row in events).items()))
    summary_by_action = dict(sorted(Counter(str(row["recommended_action"]) for row in events).items()))
    payload = {
        "version": version,
        "events": events,
        "observations": observations,
        "summary_by_code": summary_by_code,
        "summary_by_action": summary_by_action,
        "metadata": dict(metadata or {}),
    }
    return {**payload, "audit_hash": _stable_hash(payload)}


_OCCURRENCE_EVENT_CODES = frozenset({
    "detector_to_walk_drop",
    "unprofiled_profile_backed_span",
    "semantic_profile_match",
    "coarse_or_degenerate_action_menu",
    "privacy_policy_exhausted_profiled_span",
})


def _source_excerpt(document: str, start: object, end: object) -> str | None:
    try:
        left, right = int(start), int(end)
    except (TypeError, ValueError):
        return None
    if left < 0 or right <= left or right > len(document):
        return None
    before = max(document.rfind(".", 0, left), document.rfind("\n", 0, left)) + 1
    after_candidates = [index for index in (document.find(".", right), document.find("\n", right))
                        if index >= 0]
    after = min(after_candidates) + 1 if after_candidates else len(document)
    return document[before:after].strip() or None


def _ladder_diagnostic(row: Mapping[str, object], profiles: Mapping[str, object]) -> dict[str, object]:
    runtime_type = str(row.get("type", ""))
    match = row.get("match") if isinstance(row.get("match"), Mapping) else {}
    entry = match.get("entry")
    profile_rows = profiles.get("profiles", {}) if isinstance(profiles, Mapping) else {}
    profile_type = profile_rows.get(runtime_type, {}) if isinstance(profile_rows, Mapping) else {}
    profile = profile_type.get(entry, {}) if isinstance(profile_type, Mapping) else {}
    levels = _real_levels(row)
    profile_count = profile.get("count") if isinstance(profile, Mapping) else None
    level_counts = profile.get("level_counts") if isinstance(profile, Mapping) else None
    try:
        profile_count = float(profile_count)
    except (TypeError, ValueError):
        profile_count = None
    if profile_count is not None and (not math.isfinite(profile_count) or profile_count <= 0):
        profile_count = None
    counts: list[float | None] = []
    for level in levels:
        value = level_counts.get(level) if isinstance(level_counts, Mapping) else None
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = None
        counts.append(value if value is not None and math.isfinite(value) and value > 0 else None)
    jumps = [
        math.log10(right / left)
        for left, right in zip(counts, counts[1:])
        if left is not None and right is not None and left > 0 and right > 0
    ]
    first_jump = (
        math.log10(counts[0] / profile_count)
        if counts and counts[0] is not None and profile_count is not None else None
    )
    return {
        "entry": entry,
        "profile_count": profile_count,
        "levels": levels,
        "level_counts": counts,
        "level_count_complete": len(counts) == len(levels) and all(value is not None for value in counts),
        "count_monotone": all(left <= right for left, right in zip(counts, counts[1:])
                              if left is not None and right is not None),
        "first_level_log10_jump": first_jump,
        "max_adjacent_log10_jump": max(jumps) if jumps else None,
        "profile_to_coarsest_log10_span": (
            math.log10(counts[-1] / profile_count)
            if counts and counts[-1] is not None and profile_count is not None else None
        ),
    }


def _ladder_issue_codes(ladder: Mapping[str, object]) -> list[str]:
    """Return only structural ladder defects; low entry support is observational."""
    levels = ladder.get("levels")
    counts = ladder.get("level_counts")
    if not isinstance(levels, list) or len(levels) < 2:
        return ["fewer_than_two_real_levels"]
    if not ladder.get("level_count_complete"):
        return ["missing_or_invalid_level_count"]
    if not ladder.get("count_monotone"):
        return ["nonmonotone_level_counts"]
    try:
        profile_count = float(ladder.get("profile_count"))
        coarsest_count = float(counts[-1]) if isinstance(counts, list) else float("nan")
    except (TypeError, ValueError, IndexError):
        return ["missing_or_invalid_profile_count"]
    if not math.isfinite(profile_count) or profile_count <= 0.0:
        return ["missing_or_invalid_profile_count"]
    if not math.isfinite(coarsest_count) or coarsest_count <= profile_count:
        return ["no_anonymity_expansion"]
    return []


def _ladder_observation(
    *, doc_id: str, row: Mapping[str, object], ladder: Mapping[str, object]
) -> dict[str, object]:
    payload = {
        "doc_id": doc_id,
        "stage": "freeze",
        "code": "controlled_ladder_metrics",
        "entity_refs": {
            "surface": row.get("surface"), "runtime_type": row.get("type"),
            "entry": ladder.get("entry"),
        },
        "evidence": dict(ladder),
    }
    return {"observation_id": _stable_hash(payload), **payload}


def _semantic_pair_candidates(rows: list[dict[str, object]], *, embed_fn=None) -> tuple[list[dict], str | None]:
    if not rows:
        return [], None
    try:
        if embed_fn is None:
            model = _st_model(DEFAULT_MODEL_ID)
            embed_fn = lambda texts: model.encode(texts, normalize_embeddings=True)
        vectors = embed_fn([str(row["surface"]) for row in rows])
        if len(vectors) != len(rows):
            return [], "embedding_output_malformed"
        candidates: list[dict] = []
        for left_index, left in enumerate(rows):
            for right_index in range(left_index + 1, len(rows)):
                right = rows[right_index]
                if (left["doc_id"] != right["doc_id"] or
                        left["runtime_type"] != right["runtime_type"] or
                        left["entry"] == right["entry"]):
                    continue
                similarity = sum(float(a) * float(b)
                                 for a, b in zip(vectors[left_index], vectors[right_index]))
                if math.isfinite(similarity) and similarity >= SIM_FLOOR:
                    candidates.append({"left": left, "right": right,
                                       "similarity": round(similarity, 6)})
        return candidates, None
    except Exception as exc:
        return [], type(exc).__name__


def _nli_pair_relations(candidates: list[dict], *, nli_batch_fn) -> tuple[list[dict], str | None]:
    if not candidates:
        return [], None
    jobs: list[tuple[str, str, list[str]]] = []
    owners: list[tuple[dict, str]] = []
    for candidate in candidates:
        for direction, source, target in (
            ("left_implies_right", candidate["left"], candidate["right"]),
            ("right_implies_left", candidate["right"], candidate["left"]),
        ):
            context = source.get("context")
            if not isinstance(context, str) or not context:
                continue
            jobs.append((str(source["surface"]), context, [str(target["surface"])]))
            owners.append((candidate, direction))
    if not jobs:
        return [], None
    try:
        results = nli_batch_fn(jobs)
        if not isinstance(results, (list, tuple)) or len(results) != len(owners):
            return [], "nli_output_malformed"
    except Exception as exc:
        return [], type(exc).__name__
    relations: dict[tuple[str, str, str], dict] = {}
    for (candidate, direction), approved in zip(owners, results):
        if not isinstance(approved, (list, tuple)) or not approved:
            continue
        first = approved[0]
        if not isinstance(first, (list, tuple)) or len(first) != 2:
            return [], "nli_output_malformed"
        try:
            score = float(first[1])
        except (TypeError, ValueError):
            return [], "nli_output_malformed"
        if not math.isfinite(score):
            return [], "nli_output_malformed"
        left, right = candidate["left"], candidate["right"]
        key = (str(left["surface"]), str(right["surface"]), str(left["runtime_type"]))
        row = relations.setdefault(key, {**candidate, "directions": {}})
        row["directions"][direction] = score
    return list(relations.values()), None


def _controlled_self_type_scores(rows: list[dict[str, object]], *, nli_batch_fn) -> tuple[list[dict], str | None]:
    jobs, owners = [], []
    for row in rows:
        phrase = span_gate.TYPE_SUB_PHRASES.get(str(row["runtime_type"]))
        context = row.get("context")
        if phrase is None or not isinstance(context, str) or not context:
            continue
        jobs.append((str(row["surface"]), context, [phrase]))
        owners.append((row, phrase))
    if not jobs:
        return [], None
    try:
        results = nli_batch_fn(jobs)
        if not isinstance(results, (list, tuple)) or len(results) != len(owners):
            return [], "nli_output_malformed"
    except Exception as exc:
        return [], type(exc).__name__
    scored: list[dict] = []
    for (row, phrase), approved in zip(owners, results):
        score = None
        prep_filtered = not approved
        if approved:
            first = approved[0]
            if not isinstance(first, (list, tuple)) or len(first) != 2:
                return [], "nli_output_malformed"
            try:
                score = float(first[1])
            except (TypeError, ValueError):
                return [], "nli_output_malformed"
            if not math.isfinite(score):
                return [], "nli_output_malformed"
        scored.append({**row, "self_type_phrase": phrase, "self_type_score": score,
                       "prep_filtered": prep_filtered})
    return scored, None


def _coalesce_occurrence_events(events: list[dict[str, object]]) -> list[dict[str, object]]:
    """Collapse repeated mentions while retaining every source range in evidence."""
    grouped: dict[tuple[str, str, str, str, str, str], list[dict[str, object]]] = defaultdict(list)
    passthrough: list[dict[str, object]] = []
    for event in events:
        if event["code"] not in _OCCURRENCE_EVENT_CODES:
            passthrough.append(event)
            continue
        refs = dict(event.get("entity_refs") or {})
        grouped[(
            str(event["doc_id"]), str(event["stage"]), str(event["code"]),
            str(refs.get("runtime_type", "")), str(refs.get("surface", "")),
            _stable_hash(event.get("evidence") or {}),
        )].append(event)
    for rows in grouped.values():
        first = rows[0]
        refs = dict(first.get("entity_refs") or {})
        locations = [
            {"start": row.get("entity_refs", {}).get("start"),
             "end": row.get("entity_refs", {}).get("end")}
            for row in rows
            if row.get("entity_refs", {}).get("start") is not None
        ]
        refs.pop("start", None)
        refs.pop("end", None)
        evidence = dict(first.get("evidence") or {})
        evidence["occurrence_count"] = len(rows)
        if locations:
            evidence["occurrences"] = locations
        passthrough.append(_event(
            doc_id=str(first["doc_id"]), stage=str(first["stage"]), code=str(first["code"]),
            severity=str(first["severity"]), fix_class=str(first["fix_class"]),
            entity_refs=refs, evidence=evidence,
            recommended_action=str(first["recommended_action"]),
        ))
    return passthrough


def build_environment_audit(
    arms: Mapping[str, object],
    *,
    source_documents: Mapping[str, str] | None = None,
    profiles_path: str | Path | None = None,
    semantic_embed_fn=None,
    pair_nli_batch_fn=None,
    self_type_nli_batch_fn=None,
) -> dict:
    """Audit frozen detector/matcher/action-menu evidence before teacher calls.

    It intentionally reports candidates, rather than attempting profile merges or
    detector suppression.  Those are data/ontology decisions, not build-time fixes.
    """
    v2_environment_hash = None
    if arms.get("artifact_version") == "occurrence-decisions-v1":
        # V2 input adapter: manufacture the legacy audit's read-only shape from
        # canonical decisions/occurrences. No tau outcome, selected replacement,
        # risk, exhaustion, proximity, floor, or BC field crosses this boundary.
        v2_environment_hash = arms.get("environment_hash")
        v2_arms = {"_meta": {"v2_environment_hash": v2_environment_hash}}
        for doc_id, document in (arms.get("documents") or {}).items():
            decisions = {
                str(row.get("decision_id")): row
                for row in document.get("decisions") or [] if isinstance(row, Mapping)
            }
            records = []
            for occurrence in document.get("occurrences") or []:
                if not isinstance(occurrence, Mapping):
                    continue
                decision = decisions.get(str(occurrence.get("decision_id")))
                records.append({
                    "surface": occurrence.get("surface"),
                    "type": occurrence.get("runtime_type"),
                    "start": occurrence.get("start"), "end": occurrence.get("end"),
                    "uncontrolled": not bool(occurrence.get("controlled")),
                    "match": occurrence.get("match"),
                    "profile_match": occurrence.get("profile_match"),
                    "lattice": [
                        action.get("fill") for action in (decision or {}).get("actions") or []
                        if isinstance(action, Mapping) and action.get("mode") == "level"
                        and action.get("legal", True) and action.get("fill")
                    ],
                })
            v2_arms.setdefault(str(document.get("corpus", "v2")), {})[str(doc_id)] = {
                "tau_walk": ["", records],
                "detector_diagnostics": document.get("detector_diagnostics", {}),
            }
        arms = v2_arms
    events: list[dict[str, object]] = []
    observations: list[dict[str, object]] = []
    profile_path = Path(profiles_path or lp.DEFAULT_PROFILE_PATH)
    try:
        profiles = lp.load_profiles(profile_path)
    except (AttributeError, OSError, TypeError, ValueError):
        profiles = {}
    semantic_rows: list[dict[str, object]] = []
    self_type_rows: list[dict[str, object]] = []
    diagnostics_metadata: dict[str, object] = {}
    for corpus, documents in arms.items():
        if corpus == "_meta" or not isinstance(documents, Mapping):
            continue
        for doc_id, document in documents.items():
            if not isinstance(document, Mapping):
                continue
            stable_doc_id = str(doc_id)
            walk = document.get("tau_walk")
            records = walk[1] if isinstance(walk, list) and len(walk) > 1 and isinstance(walk[1], list) else []
            diagnostics = document.get("detector_diagnostics")
            diagnostics = diagnostics if isinstance(diagnostics, Mapping) else {}
            accepted = diagnostics.get("accepted") if isinstance(diagnostics.get("accepted"), list) else []
            post_rejections = (diagnostics.get("post_detection_rejections")
                               if isinstance(diagnostics.get("post_detection_rejections"), list) else [])
            post_rejection_keys = {
                _detector_row_key(row) for row in post_rejections if isinstance(row, Mapping)
            }
            by_key = {
                _detector_row_key(row): row for row in records if isinstance(row, Mapping)
            }
            source_document = (source_documents or {}).get(stable_doc_id)

            for detector_row in accepted:
                if not isinstance(detector_row, Mapping):
                    continue
                runtime_type = _detector_runtime_type(detector_row)
                if runtime_type not in PROFILE_BACKED_TYPES:
                    continue
                row = by_key.get(_detector_row_key(detector_row))
                if _detector_row_key(detector_row) in post_rejection_keys:
                    continue  # logged below as an admission decision, not a lattice abstention
                if row is None:
                    events.append(_event(
                        doc_id=stable_doc_id, stage="freeze", code="detector_to_walk_drop",
                        severity="warn", fix_class="investigate",
                        entity_refs={
                            "surface": detector_row.get("text", detector_row.get("surface")),
                            "runtime_type": runtime_type,
                            "start": detector_row.get("start"), "end": detector_row.get("end"),
                        },
                        evidence={
                            "reason": "walk_record_missing",
                            "pipeline_note": (
                                "accepted detector span has no frozen substitution record; "
                                "inspect overlap/coreference/preparation provenance"
                            ),
                        },
                        recommended_action="investigate",
                    ))
                    continue
                if row is not None and not row.get("uncontrolled"):
                    continue
                trace = row.get("profile_match") if isinstance(row, Mapping) else None
                trace = trace if isinstance(trace, Mapping) else {
                    "outcome": "not_recorded", "reason": "walk_record_missing",
                }
                events.append(_event(
                    doc_id=stable_doc_id, stage="profile_match", code="unprofiled_profile_backed_span",
                    severity="warn", fix_class="investigate",
                    entity_refs={
                        "surface": detector_row.get("text", detector_row.get("surface")),
                        "runtime_type": runtime_type,
                        "start": detector_row.get("start"), "end": detector_row.get("end"),
                    },
                    evidence={"profile_match": dict(trace), "walk_action": row.get("action") if row else None},
                    recommended_action="investigate",
                ))
                if isinstance(source_document, str):
                    self_type_rows.append({
                        "doc_id": stable_doc_id,
                        "surface": str(detector_row.get("text", detector_row.get("surface", ""))),
                        "runtime_type": runtime_type,
                        "entry": None,
                        "start": detector_row.get("start"), "end": detector_row.get("end"),
                        "context": _source_excerpt(source_document, detector_row.get("start"), detector_row.get("end")),
                    })

            controlled = [
                row for row in records if isinstance(row, Mapping)
                and row.get("lattice") and not row.get("uncontrolled")
            ]
            unique_controlled: dict[tuple[str, str], Mapping[str, object]] = {}
            for row in controlled:
                unique_controlled.setdefault((str(row.get("type", "")), _norm(row.get("surface"))), row)
                trace = row.get("profile_match")
                if isinstance(trace, Mapping) and trace.get("outcome") == "semantic":
                    events.append(_event(
                        doc_id=stable_doc_id, stage="profile_match", code="semantic_profile_match",
                        severity="info", fix_class="investigate",
                        entity_refs={"surface": row.get("surface"), "runtime_type": row.get("type"),
                                     "start": row.get("start"), "end": row.get("end")},
                        evidence={"profile_match": dict(trace)},
                        recommended_action="investigate",
                    ))
                levels = _real_levels(row)
                profile_backed = (
                    str(row.get("type", "")) in PROFILE_BACKED_TYPES
                    and isinstance(trace, Mapping)
                )
                if profile_backed and len(levels) <= 1:
                    events.append(_event(
                        doc_id=stable_doc_id, stage="freeze", code="coarse_or_degenerate_action_menu",
                        severity="warn", fix_class="data_lattice",
                        entity_refs={"surface": row.get("surface"), "runtime_type": row.get("type")},
                        evidence={"levels": levels, "action": row.get("action"),
                                  "match": row.get("match"), "profile_match": trace},
                        recommended_action="rebuild_environment",
                    ))
                if profile_backed and row.get("exhausted"):
                    events.append(_event(
                        doc_id=stable_doc_id, stage="freeze", code="privacy_policy_exhausted_profiled_span",
                        severity="info", fix_class="investigate",
                        entity_refs={"surface": row.get("surface"), "runtime_type": row.get("type")},
                        evidence={"levels": levels, "profile_match": trace},
                        recommended_action="investigate",
                    ))

            entries = list(unique_controlled.values())
            profile_decisions: dict[tuple[str, str], Mapping[str, object]] = {}
            for row in entries:
                runtime_type = str(row.get("type", ""))
                match = row.get("match") if isinstance(row.get("match"), Mapping) else {}
                entry = match.get("entry")
                if runtime_type in PROFILE_BACKED_TYPES and isinstance(entry, str) and entry:
                    profile_decisions.setdefault((runtime_type, entry), row)
            for row in profile_decisions.values():
                ladder = _ladder_diagnostic(row, profiles)
                observations.append(_ladder_observation(
                    doc_id=stable_doc_id, row=row, ladder=ladder,
                ))
                issue_codes = _ladder_issue_codes(ladder)
                if issue_codes:
                    events.append(_event(
                        doc_id=stable_doc_id, stage="freeze", code="controlled_ladder_issue",
                        severity="warn", fix_class="data_lattice",
                        entity_refs={"surface": row.get("surface"), "runtime_type": row.get("type"),
                                     "entry": ladder.get("entry")},
                        evidence={**ladder, "issues": issue_codes},
                        recommended_action="investigate",
                    ))
                if isinstance(source_document, str):
                    context = _source_excerpt(source_document, row.get("start"), row.get("end"))
                    match = row.get("match") if isinstance(row.get("match"), Mapping) else {}
                    diagnostic_row = {
                        "doc_id": stable_doc_id,
                        "surface": str(row.get("surface", "")),
                        "runtime_type": str(row.get("type", "")),
                        "entry": match.get("entry"),
                        "start": row.get("start"), "end": row.get("end"),
                        "context": context,
                    }
                    semantic_rows.append(diagnostic_row)
                    self_type_rows.append(diagnostic_row)
            for index, left in enumerate(entries):
                left_match = left.get("match") if isinstance(left.get("match"), Mapping) else {}
                for right in entries[index + 1:]:
                    if left.get("type") != right.get("type"):
                        continue
                    right_match = right.get("match") if isinstance(right.get("match"), Mapping) else {}
                    left_entry, right_entry = left_match.get("entry"), right_match.get("entry")
                    if not left_entry or not right_entry or left_entry == right_entry:
                        continue
                    if not _whole_word_contains(str(left.get("surface")), str(right.get("surface"))):
                        continue
                    event_code = "controlled_subspan_profile_conflict"
                    events.append(_event(
                        doc_id=stable_doc_id, stage="freeze", code=event_code,
                        severity="warn", fix_class="data_lattice",
                        entity_refs={
                            "left_surface": left.get("surface"), "left_profile": left_entry,
                            "right_surface": right.get("surface"), "right_profile": right_entry,
                            "runtime_type": left.get("type"),
                        },
                        evidence={"left_levels": _real_levels(left), "right_levels": _real_levels(right)},
                        recommended_action="investigate",
                    ))

            for row in post_rejections:
                if not isinstance(row, Mapping):
                    continue
                events.append(_event(
                    doc_id=stable_doc_id, stage="detector", code="post_detection_rejection",
                    severity="info", fix_class="investigate",
                    entity_refs={"surface": row.get("surface"), "runtime_type": _detector_runtime_type(row)},
                    evidence={"reason": row.get("reason"), "detector": dict(row)},
                    recommended_action="investigate",
                ))
    if source_documents is not None:
        if pair_nli_batch_fn is None or self_type_nli_batch_fn is None:
            from cloak.lattice import nli_gate_batch
            pair_nli_batch_fn = pair_nli_batch_fn or (lambda jobs: nli_gate_batch(jobs, thresh=0.6))
            self_type_nli_batch_fn = self_type_nli_batch_fn or (lambda jobs: nli_gate_batch(jobs, thresh=0.0))
        candidates, embedding_error = _semantic_pair_candidates(
            semantic_rows, embed_fn=semantic_embed_fn,
        )
        if embedding_error is not None:
            diagnostics_metadata["cross_profile_embedding_error"] = embedding_error
        relations, pair_error = _nli_pair_relations(candidates, nli_batch_fn=pair_nli_batch_fn)
        if pair_error is not None:
            diagnostics_metadata["cross_profile_nli_error"] = pair_error
        for relation in relations:
            directions = relation["directions"]
            relation_kind = (
                "equivalent_candidate" if len(directions) == 2 else "specific_to_general_candidate"
            )
            left, right = relation["left"], relation["right"]
            events.append(_event(
                doc_id=str(left["doc_id"]), stage="freeze", code="cross_profile_coreference_candidate",
                severity="warn", fix_class="investigate",
                entity_refs={
                    "runtime_type": left["runtime_type"],
                    "left_surface": left["surface"], "left_profile": left["entry"],
                    "right_surface": right["surface"], "right_profile": right["entry"],
                },
                evidence={
                    "candidate_kind": relation_kind,
                    "embedding_similarity": relation["similarity"],
                    "directional_entailment": directions,
                    "left_excerpt": left.get("context"), "right_excerpt": right.get("context"),
                },
                recommended_action="investigate",
            ))
        self_scores, self_type_error = _controlled_self_type_scores(
            self_type_rows, nli_batch_fn=self_type_nli_batch_fn,
        )
        if self_type_error is not None:
            diagnostics_metadata["controlled_self_type_nli_error"] = self_type_error
        for score in self_scores:
            diagnostic_code = (
                "unprofiled_self_type_diagnostic"
                if score.get("entry") is None else "controlled_self_type_diagnostic"
            )
            events.append(_event(
                doc_id=str(score["doc_id"]), stage="detector", code=diagnostic_code,
                severity="info", fix_class="investigate",
                entity_refs={"surface": score["surface"], "runtime_type": score["runtime_type"],
                             "entry": score["entry"], "start": score["start"], "end": score["end"]},
                evidence={"self_type_phrase": score["self_type_phrase"],
                          "self_type_score": score["self_type_score"],
                          "prep_filtered": score["prep_filtered"],
                          "source_excerpt": score.get("context")},
                recommended_action="investigate",
            ))
    return _finalize(
        ENVIRONMENT_AUDIT_VERSION,
        _coalesce_occurrence_events(events),
        observations=observations,
        metadata={
        "arms_meta": dict(arms.get("_meta") or {}) if isinstance(arms.get("_meta"), Mapping) else {},
        "environment_hash": v2_environment_hash,
        "profiles_path": str(profile_path),
        "nli_diagnostics_enabled": source_documents is not None,
        **diagnostics_metadata,
        },
    )


def build_legacy_environment_audit(arms: Mapping[str, object], **kwargs) -> dict:
    """Explicit historical-arms audit compatibility entrypoint."""
    return build_environment_audit(arms, **kwargs)


def build_qa_audit(artifact: Mapping[str, object], *, environment_audit: Mapping[str, object] | None = None) -> dict:
    """Normalize the QA build lifecycle into actionable, per-document events."""
    events: list[dict[str, object]] = []
    for doc_id, flags in (artifact.get("review_flags") or {}).items():
        if not isinstance(flags, list):
            continue
        for flag in flags:
            if not isinstance(flag, Mapping):
                continue
            events.append(_event(
                doc_id=str(doc_id), stage=str(flag.get("stage", "review")),
                code=str(flag.get("code", "review_flag")), severity=str(flag.get("severity", "info")),
                fix_class=str(flag.get("fix_class", "investigate")), evidence=dict(flag.get("detail") or {}),
                recommended_action=_ACTION_FOR_FIX_CLASS.get(
                    str(flag.get("fix_class", "")), "investigate"
                ),
            ))
    for record in (artifact.get("rejections") or {}).get("records", []):
        if not isinstance(record, Mapping):
            continue
        doc_id = str(record.get("doc_id", "unknown"))
        reason = str(record.get("detail_reason") or record.get("reason") or "rejected")
        evidence = record.get("evidence") if isinstance(record.get("evidence"), Mapping) else {}
        action = "teacher_repair" if reason in {
            "teacher_abstained", "not_generated", "generation_failed", "invalid_evidence",
            "invalid_question", "invalid_property", "answer_leakage", "literal_will_be_substituted",
        } else "investigate"
        if reason in {"three_point_gate_failed", "reader_unstable", "context_reader_failed"}:
            action = "reader_regate"
        events.append(_event(
            doc_id=doc_id, stage="reader_gate" if reason == "three_point_gate_failed" else "compile",
            code=f"relation_rejected:{reason}", severity="warn", fix_class=action,
            entity_refs={"relation": record.get("relation"), "question": record.get("question")},
            evidence={"reason": reason, "attempt": record.get("attempt"), "evidence": evidence},
            recommended_action=action,
        ))
        for diagnostic in evidence.get("anchor_diagnostics") or []:
            if not isinstance(diagnostic, Mapping) or diagnostic.get("kind") != "soft_cross_clause_cap_exceeded":
                continue
            events.append(_event(
                doc_id=doc_id, stage="reader_gate", code="soft_cross_clause_cap_exceeded",
                severity="warn", fix_class="investigate",
                entity_refs={"relation": record.get("relation"), "question": record.get("question")},
                evidence=dict(diagnostic), recommended_action="investigate",
            ))
    for doc_id, attempts in (artifact.get("relation_generation") or {}).items():
        for attempt in attempts if isinstance(attempts, list) else []:
            if not isinstance(attempt, Mapping):
                continue
            status = str(attempt.get("status", "unknown"))
            events.append(_event(
                doc_id=str(doc_id), stage="teacher", code="teacher_relation_attempt",
                severity="info" if status == "kept" else "warn",
                fix_class="teacher_repair" if status != "kept" else "investigate",
                entity_refs={"relation": attempt.get("relation"),
                             "proposal_index": attempt.get("proposal_index"),
                             "teacher_id": attempt.get("teacher_id"),
                             "run_id": attempt.get("run_id")},
                evidence={
                    "status": status, "reason": attempt.get("reason"),
                    "arguments": attempt.get("arguments"), "question": attempt.get("question"),
                    "accepted_answers": attempt.get("accepted_answers"),
                },
                recommended_action="teacher_repair" if status != "kept" else "investigate",
            ))
    for doc_id, runs in (artifact.get("relation_teacher_runs") or {}).items():
        for run in runs if isinstance(runs, list) else []:
            if not isinstance(run, Mapping):
                continue
            events.append(_event(
                doc_id=str(doc_id), stage="teacher", code="teacher_run",
                severity="info" if run.get("status") in {"proposed", "kept"} else "warn",
                fix_class="teacher_repair" if run.get("status") in {"failed", "abstained"} else "investigate",
                entity_refs={"teacher_id": run.get("teacher_id"), "run_id": run.get("run_id")},
                evidence=dict(run),
                recommended_action="teacher_repair" if run.get("status") in {"failed", "abstained"}
                else "investigate",
            ))
    for assertion in (artifact.get("assertions") or {}).values():
        if not isinstance(assertion, Mapping):
            continue
        evidence = assertion.get("evidence") if isinstance(assertion.get("evidence"), Mapping) else {}
        for diagnostic in evidence.get("anchor_diagnostics") or []:
            if not isinstance(diagnostic, Mapping) or diagnostic.get("kind") != "soft_cross_clause_cap_exceeded":
                continue
            events.append(_event(
                doc_id=str(assertion.get("doc_id")), stage="compile", code="soft_cross_clause_cap_exceeded",
                severity="warn", fix_class="investigate",
                entity_refs={"relation": assertion.get("relation"), "question": assertion.get("question")},
                evidence=dict(diagnostic), recommended_action="investigate",
            ))
    for doc_id, rows in (artifact.get("relation_candidate_accounting") or {}).items():
        for row in rows if isinstance(rows, list) else []:
            if isinstance(row, Mapping) and row.get("state") in {"ledger_inconsistent", "missed"}:
                events.append(_event(
                    doc_id=str(doc_id), stage="teacher", code=f"teacher_candidate:{row.get('state')}",
                    severity="warn", fix_class="teacher_repair", evidence=dict(row),
                    recommended_action="teacher_repair",
                ))
    for doc_id, row in (artifact.get("relation_gleaning") or {}).items():
        if not isinstance(row, Mapping):
            continue
        if row.get("triggered") and row.get("returned_count", 0) == 0:
            events.append(_event(
                doc_id=str(doc_id), stage="teacher", code="gleaning_returned_no_relations",
                severity="warn", fix_class="teacher_repair", evidence=dict(row),
                recommended_action="teacher_repair",
            ))
    for doc_id, coverage in (artifact.get("relation_coverage") or {}).items():
        if not isinstance(coverage, Mapping):
            continue
        for target in coverage.get("unresolved_targets") or []:
            if not isinstance(target, Mapping):
                continue
            kind = str(target.get("kind", "unknown"))
            code = {
                "missed": "teacher_missed_structural_opportunity",
                "fixable": "repairable_relation_still_rejected",
                "ambiguous": "relation_answer_ambiguity_requires_review",
            }.get(kind, "unresolved_relation_target")
            action = "teacher_repair" if kind in {"missed", "fixable"} else "manual_review"
            events.append(_event(
                doc_id=str(doc_id), stage="coverage", code=code,
                severity="warn" if kind != "ambiguous" else "info", fix_class=action,
                entity_refs={"relation": target.get("relation"),
                             "fact_key_hash": target.get("fact_key_hash")},
                evidence=dict(target), recommended_action=action,
            ))
    metadata: dict[str, object] = {}
    if environment_audit is not None:
        metadata["environment_audit_hash"] = environment_audit.get("audit_hash")
    return _finalize(QA_AUDIT_VERSION, events, metadata=metadata)


def audit_markdown(audit: Mapping[str, object]) -> str:
    lines = [
        f"# {audit.get('version', 'qa-audit')}", "",
        f"- Audit hash: `{audit.get('audit_hash', '')}`",
        f"- Events: `{len(audit.get('events') or [])}`", "",
        f"- Passive observations: `{len(audit.get('observations') or [])}`", "",
        "## Summary", "",
    ]
    for code, count in (audit.get("summary_by_code") or {}).items():
        lines.append(f"- `{code}`: {count}")
    events_by_doc: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for event in audit.get("events") or []:
        if isinstance(event, Mapping):
            events_by_doc[str(event.get("doc_id"))].append(event)
    for doc_id, events in sorted(events_by_doc.items()):
        lines.extend(["", f"## {doc_id}", ""])
        for event in events:
            lines.append(
                f"- **{event.get('severity')}** `{event.get('code')}` → "
                f"`{event.get('recommended_action')}` (`{event.get('event_id')}`)"
            )
            evidence = event.get("evidence")
            if evidence:
                lines.append(f"  - Evidence: `{json.dumps(evidence, sort_keys=True, default=str)}`")
    return "\n".join(lines) + "\n"


def write_audit_sidecars(audit: Mapping[str, object], output: Path) -> tuple[Path, Path, Path]:
    """Write full JSON, typed event/observation JSONL, and a readable Markdown report."""
    # ``output`` is a logical report stem and may itself contain dot-separated
    # labels (for example ``artifact.arms.environment-audit``). Appending avoids
    # replacing the source artifact's suffix and accidentally overwriting it.
    json_path = output.with_name(output.name + ".json")
    jsonl_path = output.with_name(output.name + ".jsonl")
    markdown_path = output.with_name(output.name + ".md")
    json_path.write_text(json.dumps(audit, indent=1, default=str))
    jsonl_records = (
        [{"record_kind": "event", **event} for event in audit.get("events") or []]
        + [{"record_kind": "observation", **observation}
           for observation in audit.get("observations") or []]
    )
    jsonl_path.write_text("".join(
        json.dumps(record, sort_keys=True, default=str) + "\n"
        for record in jsonl_records
    ))
    markdown_path.write_text(audit_markdown(audit))
    return json_path, jsonl_path, markdown_path
