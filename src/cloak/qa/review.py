"""Build-report review flags over a packaged utility artifact."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping

from cloak.qa.builder import (
    GLEANING_FIXABLE_REASONS,
    GLEANING_FIX_HINTS,
    ordered_decision_levels,
    relation_scope,
)
from cloak.qa.scoring import canon
from cloak.qa.teacher import RELATION_TEACHER_MAX_RELATIONS

# Detected runtime types that are supposed to carry a generalization lattice
# (identity types like PERSON/LOC are placeholder-by-rule and excluded).
_REVIEW_LADDER_TYPES = frozenset({"drug", "health-condition", "medical-procedure"})


def _review_flag(code: str, stage: str, fix_class: str, severity: str, detail: dict) -> dict:
    return {"code": code, "stage": stage, "fix_class": fix_class,
            "severity": severity, "detail": detail}


def compute_review_flags(artifact: Mapping) -> dict[str, list[dict]]:
    """Per-document diagnostics classifying *why* a document may be worth
    re-processing after a fix. Pure function over the built artifact; each flag
    carries a `fix_class` (data_lattice / teacher_redraw / reader /
    ontology_review) so a change can re-select exactly the affected documents.
    Diagnostic only — never enters the artifact hash or any measurement."""
    flags: dict[str, list[dict]] = defaultdict(list)

    # A) a detected, lattice-eligible span with no legal generalization level
    #    (e.g. an unaliased drug brand -> placeholder-only, the synthroid case).
    for doc_id, document in (artifact.get("documents") or {}).items():
        decisions = {str(d.get("decision_id")): d for d in document.get("decisions") or []}
        seen: set[tuple[str, str]] = set()
        for occurrence in document.get("occurrences") or []:
            runtime_type = canon(str(occurrence.get("runtime_type", "")))
            if runtime_type not in _REVIEW_LADDER_TYPES:
                continue
            key = (runtime_type, canon(str(occurrence.get("surface", ""))))
            if key in seen:
                continue
            seen.add(key)
            decision = decisions.get(str(occurrence.get("decision_id")))
            if not (ordered_decision_levels(decision) if decision else []):
                flags[doc_id].append(_review_flag(
                    "missing_generalization", "freeze", "data_lattice", "warn",
                    {"runtime_type": occurrence.get("runtime_type"),
                     "surface": occurrence.get("surface"),
                     "resolved_decision": bool(decision)}))

    # C) soft relation-count cap: the hard K-cap reject was removed, so a relation-dense note
    #    keeps all its valid relations. Log (info, no fix_class action) when a document's kept
    #    relation count exceeds the soft cap, so unusually dense notes are still visible.
    for doc_id, records in (artifact.get("relation_generation") or {}).items():
        kept = sum(1 for record in records if record.get("status") == "kept")
        if kept > RELATION_TEACHER_MAX_RELATIONS:
            flags[str(doc_id)].append(_review_flag(
                "relation_count_over_soft_cap", "compile", "ontology_review", "info",
                {"kept_relations": kept, "soft_cap": RELATION_TEACHER_MAX_RELATIONS}))

    # False-positive guard for the lattice_level_suspect probe: a lattice node used by a KEPT relation
    # in the same doc is provably readable, so a gate failure elsewhere is context-specific, not bad
    # data. Map each doc's kept-assertion decisions back to their canonical surfaces.
    _surface_by_decision = {
        str(doc_id): {
            str(d.get("decision_id")): canon(str(d.get("canonical_key") or ""))
            for d in (document.get("decisions") or [])
        }
        for doc_id, document in (artifact.get("documents") or {}).items()
    }
    kept_lattice_surfaces: dict[str, set[str]] = defaultdict(set)
    for assertion in (artifact.get("assertions") or {}).values():
        _dec_surface = _surface_by_decision.get(str(assertion.get("doc_id")), {})
        for decision_id in (assertion.get("decision_requirements") or {}):
            surface = _dec_surface.get(str(decision_id))
            if surface:
                kept_lattice_surfaces[str(assertion.get("doc_id"))].add(surface)

    # D) classify signals the build already emits
    for record in (artifact.get("rejections") or {}).get("records") or []:
        doc_id = str(record.get("doc_id"))
        reason = record.get("detail_reason") or record.get("reason")
        evidence = record.get("evidence") or {}
        if reason == "literal_will_be_substituted":
            flags[doc_id].append(_review_flag(
                "literal_will_be_substituted", "compile", "data_lattice", "warn", {}))
        elif reason == "answer_leakage":
            flags[doc_id].append(_review_flag(
                "unrepaired_answer_leakage", "compile", "ontology_review", "warn",
                {"leaking_tokens": evidence.get("leaking_tokens"),
                 "answer_levels": evidence.get("answer_levels")}))
        elif reason == "placeholder_answerable":
            flags[doc_id].append(_review_flag(
                "placeholder_answerable", "gate", "ontology_review", "info", {}))
        elif reason == "three_point_gate_failed":
            scores = (evidence.get("validation") or {}).get("scores") or {}
            probe = evidence.get("lattice_probe")
            if isinstance(probe, Mapping) and probe.get("readable_coarser_level"):
                # a COARSER legal level in the same chain reads -> the chosen fine level, not the
                # relation, could be bad lattice data. But this probe is FALSE-POSITIVE-PRONE: the
                # failure is usually ANSWER AMBIGUITY (subject has >=2 same-type answers -> no unique
                # QA, not a bad level), or the flagged node is used by a KEPT relation in-doc (provably
                # readable). Suppress both; only a node that fails AND is unambiguous AND is never kept
                # in-doc is a genuine data suspect worth a human fixing lattice_profiles.json.
                if evidence.get("answer_competing") or (
                        canon(str(probe.get("surface", "")))
                        in kept_lattice_surfaces.get(doc_id, set())):
                    pass
                else:
                    flags[doc_id].append(_review_flag(
                        "lattice_level_suspect", "gate", "data_lattice", "warn",
                        {"surface": probe.get("surface"),
                         "runtime_type": probe.get("runtime_type"),
                         "unreadable_level": probe.get("unreadable_level"),
                         "readable_coarser_level": probe.get("readable_coarser_level"),
                         "chain": probe.get("chain"),
                         "scores": scores}))
            elif (scores.get("original", 0.0) >= 1.0
                    and scores.get("representative", 0.0) < 1.0
                    and scores.get("placeholder", 1.0) < 1.0):
                # source answerable, generalized form not -> the chosen generalization level is
                # not reader-recoverable. Treated as a lattice-profile signal (level too coarse /
                # mislabeled for the surface). NOTE: a minority of these are genuine reader limits
                # rather than data; the `scores` detail lets a human triage.
                flags[doc_id].append(_review_flag(
                    "representative_unreadable", "gate", "data_lattice", "warn",
                    {"scores": scores}))
            elif (scores.get("original", 1.0) < 1.0
                    and scores.get("representative", 0.0) >= 1.0):
                # answer recoverable ONLY when the level is literally rendered (generalized
                # render passes, source render does not) -> the accepted level is not
                # reader-recoverable from the source surface. Strong lattice-profile signal:
                # missing/insufficient answer_aliases or a level too abstract for the surface.
                flags[doc_id].append(_review_flag(
                    "answer_only_readable_when_generalized", "gate", "data_lattice", "warn",
                    {"scores": scores, "subtype": evidence.get("subtype"),
                     "occurrence_ids": evidence.get("occurrence_ids")}))

    # D) a repair that had to lower the answer floor is worth a human look
    for row in (artifact.get("assertions") or {}).values():
        repair = (row.get("evidence") or {}).get("leakage_repair")
        if isinstance(repair, Mapping) and repair.get("floor_lowered"):
            flags[str(row.get("doc_id"))].append(_review_flag(
                "floor_lowered_repair", "compile", "ontology_review", "info",
                {"from_level": repair.get("from_level"), "to_level": repair.get("to_level")}))
        # Hedge/modality observation (non-blocking): the source is conditional/planned but the
        # question is definite -- surfaced for review, the relation was NOT blocked from the reader.
        modality = (row.get("evidence") or {}).get("modality_diagnostics")
        if modality:
            flags[str(row.get("doc_id"))].append(_review_flag(
                "hedged_source_definite_question", "compile", "ontology_review", "info",
                {"diagnostics": list(modality)}))

    # ambiguity MONITOR: a relation whose answered type has >1 co-valid answer in scope. The
    # free-form reader can name a different (also-valid) one -> non-deterministic utility signal.
    # Diagnostic only (info); reported for kept relations and gate-rejected ones alike.
    for row in (artifact.get("assertions") or {}).values():
        competing = (row.get("evidence") or {}).get("answer_competing")
        if competing:
            flags[str(row.get("doc_id"))].append(_review_flag(
                "relation_answer_ambiguous", "compile", "ambiguity", "info",
                {"relation": row.get("relation"), "competing_answers": competing}))
    for record in (artifact.get("rejections") or {}).get("records") or []:
        competing = (record.get("evidence") or {}).get("answer_competing")
        if competing:
            flags[str(record.get("doc_id"))].append(_review_flag(
                "relation_answer_ambiguous", "gate", "ambiguity", "info",
                {"detail_reason": record.get("detail_reason"), "competing_answers": competing}))

    # teacher self-report inconsistency -> the draw is unreliable, re-run it
    for doc_id, rows in (artifact.get("relation_candidate_accounting") or {}).items():
        if any(isinstance(r, Mapping) and r.get("state") == "ledger_inconsistent" for r in rows):
            flags[doc_id].append(_review_flag(
                "ledger_inconsistent", "compile", "teacher_redraw", "warn", {}))

    # teacher-repairable but unresolved: a relation whose only outcomes are FIXABLE-reason
    # rejections (no kept sibling for the same fact) that the repair pass did not rescue -- either
    # the secondary teacher abstained, or it re-proposed and the fix still rejects. These are the
    # relations a better teacher draw could recover; deterministic rewriting is deliberately NOT
    # attempted (surface recolor already ran in substitute_linked_surfaces; the residue is
    # un-substitutable source text). fix_class teacher_redraw -> re-selectable for a repair re-run.
    def _fact_key(row: Mapping) -> tuple:
        parts = []
        for argument in row.get("arguments") or []:
            parts.append((
                argument.get("role"),
                argument.get("span_label") or argument.get("literal")
                or argument.get("support_property"),
            ))
        return (row.get("relation"), tuple(parts))

    gleaning = artifact.get("relation_gleaning") or {}
    for doc_id, rows in (artifact.get("relation_generation") or {}).items():
        kept_keys = {_fact_key(r) for r in rows if r.get("status") == "kept"}
        by_key: dict[tuple, list[Mapping]] = defaultdict(list)
        for r in rows:
            if r.get("status") == "rejected" and r.get("reason") in GLEANING_FIXABLE_REASONS:
                by_key[_fact_key(r)].append(r)
        abstained = str((gleaning.get(doc_id) or {}).get("secondary_status")) == "abstained"
        for key, rejected in by_key.items():
            if key in kept_keys:
                continue  # recovered elsewhere -> not an unresolved loss
            run_ids = {r.get("run_id") for r in rejected}
            if "gleaning" in run_ids:
                disposition = "unresolved_after_repair"  # C2 re-proposed, still rejects
            elif abstained:
                disposition = "repair_abstained"          # C2 declined to re-author it
            else:
                disposition = "repair_not_returned"        # C2 ran but did not target it
            latest = rejected[-1]
            try:
                scope = relation_scope(latest.get("arguments") or [])
            except ValueError:
                scope = None
            flags[doc_id].append(_review_flag(
                "teacher_repairable_unresolved", "gate", "teacher_redraw", "warn",
                {"relation": latest.get("relation"),
                 "reason": latest.get("reason"),
                 "hint": GLEANING_FIX_HINTS.get(str(latest.get("reason"))),
                 "disposition": disposition,
                 "scope": scope}))

    return {doc_id: doc_flags for doc_id, doc_flags in flags.items()}
