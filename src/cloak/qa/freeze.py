"""Frozen ranker-environment production, the legacy-arms bridge, and migrations."""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from math import isfinite
from numbers import Real

from cloak.runtime_types import placeholder_token, placeholder_type_token
from cloak.qa.scoring import canon, normalized_aliases, stable_hash

def _frozen_semantic_chain(decision: Mapping, source_aliases: Sequence[str]) -> list[dict]:
    """Freeze explicit entailment closure for one decision's lattice.

    Ordered specific -> coarse: KEEP entails every level; a level entails itself
    and every coarser level; placeholder entails nothing. This is a declaration
    by the accepted lattice profile, not proven natural-language entailment; the
    linked-answer scorer reads it instead of re-deriving from mutable profiles,
    counts, or lexical similarity (docs/handoffs/2026-07-14-qa-reader-lattice-scoring.md).
    """
    # Specific -> coarse order is the decision's AUTHORED level ladder (the profile's `levels`
    # list order, preserved into `actions` at freeze time). The profile loader validates that
    # ladder is monotone in the profile's OWN level_counts (lattice/profiles.py), so authored
    # order == profile-count order and is the semantic hierarchy.
    #
    # We deliberately do NOT re-sort by `coarseness_rank`: that is a GLOBAL per-fill anonymity-set
    # size (aset_count -> lookup_count, keyed on the level string across ALL profiles, aggregated
    # by max), not a per-profile generality rank. It is miscalibrated across profiles -- e.g.
    # global "heart disease"=400 > "thoracic disease"=390 -- so sorting by it inverts organ vs
    # region levels (heart ranked coarser than thoracic), which drops the organ->region entailment
    # edge and inverts the RL reward gradient (a policy generalizing CHF to the more-specific
    # "heart disease" would score 0 while the coarser "thoracic disease" scores 1). aset /
    # coarseness_rank is UNCHANGED for its real jobs (ranker policy feature, BC target selection,
    # hiding / representative-anchor selection); only the entailment ORDER uses the profile ladder.
    levels = [action for action in decision.get("actions", []) if action.get("mode") == "level"]
    ordered_properties = [
        canon(str(action["entails"][0]))
        for action in levels
        if action.get("entails")
    ]
    canonical_key = str(decision.get("canonical_key", ""))
    keep_aliases = list(dict.fromkeys(
        [canonical_key, *[str(alias) for alias in source_aliases]]
    ))
    chain = [{
        "node": "keep",
        "answer_aliases": [alias for alias in keep_aliases if alias],
        "entailed_properties": list(ordered_properties),
    }]
    for index, prop in enumerate(ordered_properties):
        chain.append({
            "node": prop,
            "answer_aliases": [prop],
            "entailed_properties": ordered_properties[index:],
        })
    chain.append({"node": "placeholder", "answer_aliases": [], "entailed_properties": []})
    return chain


def _entity_key(surface: str, runtime_type: str) -> str:
    """Canonical identity for a detected surface, shared by decision keys and occurrence
    matching so plural/alias variants ("migraines"->"migraine", brand->generic) resolve to
    one decision. Reuses the lattice profile resolver (single source of the plural fold);
    falls back to canon() for surfaces with no profile row."""
    from cloak.lattice.profiles import lookup_entry
    entry = lookup_entry(surface, runtime_type)
    return canon(entry[0]) if entry else canon(surface)


def _append_text_anchored_occurrences(
    document_text: str, doc_id: str,
    decisions_by_key: Mapping[tuple[str, str], dict], occurrences: list[dict],
) -> None:
    """Register exact-text occurrences of already-decided entities that the detector
    dropped locally -- e.g. a repeat mention scored below the per-type admission gate
    ('kidney stones' at 0.44 vs the 0.5 health-condition gate) even though the same
    surface was admitted >=gate elsewhere in the document. Grounding can then anchor a
    relation argument to the dropped mention.

    Threshold-free: only surfaces that already back a *controlled decision in this
    document* are propagated, so no new entity type or value is introduced -- the value
    is confirmed sensitive; we only recover its other verbatim positions. Positions
    already covered by an occurrence are skipped, so detected spans are never duplicated.
    One compiled regex + one finditer pass over the document -> O(len(document)); the
    per-match overlap test is O(#occurrences), tiny for a single note.
    """
    from cloak.lattice.profiles import singularize
    if not document_text:
        return
    base_to_decision: dict[str, dict] = {}
    decisions_by_id = {d["decision_id"]: d for d in decisions_by_key.values()}
    for occ in occurrences:
        decision_id = occ.get("decision_id")
        if not occ.get("controlled") or decision_id is None:
            continue
        base = singularize(str(occ.get("surface", "")))
        if len(base) >= 3:  # ponytail: skip 1-2 char bases ("ms","mi") -- match everything
            base_to_decision.setdefault(base, decisions_by_id[decision_id])
    if not base_to_decision:
        return
    # longest base first so "kidney stone" wins over "stone"; trailing s? folds plurals,
    # hyphen/word guards keep "ct" out of "contract" and "stone" out of "stoned".
    bases = sorted(base_to_decision, key=len, reverse=True)
    pattern = re.compile(
        r"(?<![\w-])(" + "|".join(re.escape(b) for b in bases) + r")s?(?![\w-])",
        re.IGNORECASE,
    )
    covered = [
        (int(o["start"]), int(o["end"])) for o in occurrences
        if isinstance(o.get("start"), int) and isinstance(o.get("end"), int)
    ]
    for match in pattern.finditer(document_text):
        start, end = match.start(), match.end()
        if any(start < ce and cs < end for cs, ce in covered):
            continue
        decision = base_to_decision.get(singularize(match.group(1)))
        if decision is None:
            continue
        surface = document_text[start:end]
        occurrence_id = stable_hash({
            "doc_id": doc_id, "runtime_type": decision["runtime_type"],
            "surface": surface, "start": start, "end": end,
        })
        occurrences.append({
            "occurrence_id": occurrence_id,
            "start": start, "end": end, "surface": surface, "aliases": [],
            "runtime_type": decision["runtime_type"], "polarity": "unknown",
            "detector_provenance": {
                "source": "text_anchored", "anchored_to": decision["decision_id"]},
            "overlap_disposition": "accepted",
            "decision_id": decision["decision_id"], "controlled": True,
        })
        decision["occurrence_ids"].append(occurrence_id)
        covered.append((start, end))


def freeze_ranker_environment(
    ranker_environment: Mapping,
    *,
    occurrences_by_document: Mapping[str, Sequence[Mapping]] | None = None,
    source_documents: Mapping[str, str] | None = None,
) -> dict:
    """Migrate embedded ranker spans to stable occurrence/decision identities, without detection.

    When ``source_documents`` is provided, exact-text repeats of an already-decided entity
    that the detector dropped locally are recovered as occurrences (see
    ``_append_text_anchored_occurrences``). Both callers pass the same source text so the
    ``environment_hash`` stays identical between build and train.
    """
    documents: dict[str, dict] = {}
    for corpus, per_document in ranker_environment.get("corpora", {}).items():
        for doc_id, document in per_document.items():
            decisions_by_key: dict[tuple[str, str], dict] = {}
            for span in document.get("spans", []):
                runtime_type = str(span.get("type", ""))
                surface = str(span.get("surface", ""))
                decision_key = (runtime_type, canon(surface))
                decision_id = stable_hash({
                    "doc_id": doc_id,
                    "runtime_type": runtime_type,
                    "canonical_surface": decision_key[1],
                })
                actions = []
                action_ids = set()
                for action in span.get("actions", []):
                    declared_mode = str(action.get("mode", "level"))
                    if declared_mode not in {"level", "keep", "placeholder"}:
                        raise ValueError(
                            f"invalid action mode for decision {decision_id}: {declared_mode}"
                        )
                    fill = action.get("fill")
                    if declared_mode == "placeholder" and action.get("keep"):
                        raise ValueError(
                            f"placeholder action cannot be KEEP for decision {decision_id}"
                        )
                    source_fill = bool(fill) and canon(str(fill)) == decision_key[1]
                    is_keep = bool(action.get("keep")) or declared_mode == "keep" or (
                        declared_mode == "level" and source_fill
                    )
                    if action.get("source_identity") and not is_keep:
                        raise ValueError(
                            f"source_identity is reserved for KEEP actions on {decision_id}"
                        )
                    if is_keep and not source_fill:
                        raise ValueError(
                            f"KEEP action must preserve the source identity for {decision_id}"
                        )
                    mode = (
                        "keep" if is_keep else
                        "placeholder" if declared_mode == "placeholder" else
                        "level"
                    )
                    action_semantics = {
                        **dict(action),
                        "mode": mode,
                        "fill": decision_key[1] if is_keep else fill,
                        "legal": bool(action.get("legal", True)),
                    }
                    action_semantics.pop("action_id", None)
                    if mode == "keep":
                        action_semantics.update({
                            "keep": True,
                            "source_identity": True,
                            "entails": [decision_key[1]],
                        })
                        action_semantics.pop("coarseness_rank", None)
                    elif mode == "placeholder":
                        action_semantics["entails"] = []
                        action_semantics.pop("coarseness_rank", None)
                    else:
                        rank = action.get("coarseness_rank", action.get("aset"))
                        if action_semantics["legal"] and (
                            isinstance(rank, bool)
                            or not isinstance(rank, Real)
                            or not isfinite(float(rank))
                        ):
                            raise ValueError(
                                "legal level action requires numeric coarseness_rank "
                                f"or aset for decision {decision_id}"
                            )
                        if isinstance(rank, Real) and not isinstance(rank, bool):
                            action_semantics["coarseness_rank"] = rank
                        action_semantics["entails"] = (
                            [canon(str(fill))] if fill else []
                        )
                    action_id = stable_hash({
                        "decision_id": decision_id,
                        "action": action_semantics,
                    })
                    if action_id in action_ids:
                        raise ValueError(f"duplicate action semantics for decision {decision_id}")
                    action_ids.add(action_id)
                    actions.append({
                        **action_semantics,
                        "action_id": action_id,
                    })
                forced_placeholder = any(
                    action.get("mode") == "placeholder" and action.get("forced_placeholder")
                    for action in actions
                )
                if not forced_placeholder and not any(action["mode"] == "keep" for action in actions):
                    keep_semantics = {
                        "fill": decision_key[1],
                        "mode": "keep",
                        "keep": True,
                        "source_identity": True,
                        "legal": True,
                        "entails": [decision_key[1]],
                    }
                    keep_action = {
                        **keep_semantics,
                        "action_id": stable_hash({
                            "decision_id": decision_id,
                            "action": keep_semantics,
                        }),
                    }
                    placeholder_index = next(
                        (index for index, action in enumerate(actions)
                         if action["mode"] == "placeholder"),
                        len(actions),
                    )
                    actions.insert(placeholder_index, keep_action)
                action_menu_hash = stable_hash(actions)
                previous = decisions_by_key.get(decision_key)
                if previous and previous["action_menu_hash"] != action_menu_hash:
                    raise ValueError(
                        f"inconsistent action menus for repeated decision {decision_id}"
                    )
                if previous is None:
                    decisions_by_key[decision_key] = {
                        "decision_id": decision_id,
                        "runtime_type": runtime_type,
                        "canonical_key": decision_key[1],
                        "occurrence_ids": [],
                        "controlled": True,
                        "ranker_selectable": not forced_placeholder,
                        "actions": actions,
                        "action_menu_hash": action_menu_hash,
                    }
            # KEEP action semantics intentionally remain keyed by each source surface
            # above. Once those menus are complete, aliases that resolve to the same
            # lattice entry can share the first decision -- but only when their
            # non-KEEP choices are indistinguishable. Redirecting the surface keys
            # preserves the KEEP construction invariant while making every occurrence
            # receive the shared decision_id below.
            def non_keep_menu(decision: Mapping) -> tuple:
                entries = []
                for action in decision["actions"]:
                    if action["mode"] == "keep":
                        continue
                    if action["mode"] == "level":
                        entries.append((
                            "level", action.get("fill"),
                            float(action["coarseness_rank"]),
                        ))
                    else:
                        entries.append(("placeholder", action.get("fill")))
                return tuple(sorted(entries))

            keys_by_entity: dict[tuple[str, str], list[tuple[str, str]]] = {}
            for decision_key in decisions_by_key:
                runtime_type, source_key = decision_key
                keys_by_entity.setdefault(
                    (runtime_type, _entity_key(source_key, runtime_type)), []
                ).append(decision_key)
            for decision_keys in keys_by_entity.values():
                primary = decisions_by_key[decision_keys[0]]
                primary_menu = non_keep_menu(primary)
                if all(
                    non_keep_menu(decisions_by_key[decision_key]) == primary_menu
                    for decision_key in decision_keys[1:]
                ):
                    for decision_key in decision_keys[1:]:
                        decisions_by_key[decision_key] = primary
            occurrence_source = (
                occurrences_by_document[doc_id]
                if occurrences_by_document is not None and doc_id in occurrences_by_document
                else document.get("spans", [])
            )
            # Secondary index by profile-canonical identity so a plural/alias occurrence
            # ("migraines", a brand name) still links to its decision even though the
            # decision key stays the source surface (which KEEP semantics rely on).
            decisions_by_entity: dict[tuple[str, str], dict] = {}
            for (rtype, ckey), decision in decisions_by_key.items():
                decisions_by_entity.setdefault((rtype, _entity_key(ckey, rtype)), decision)
            occurrences = []
            for row in occurrence_source:
                runtime_type = str(row.get("type", row.get("runtime_type", "")))
                surface = str(row.get("surface", ""))
                decision = (
                    decisions_by_key.get((runtime_type, canon(surface)))
                    or decisions_by_entity.get((runtime_type, _entity_key(surface, runtime_type)))
                )
                occurrence_id = stable_hash({
                    "doc_id": doc_id,
                    "runtime_type": runtime_type,
                    "surface": surface,
                    "start": row.get("start"),
                    "end": row.get("end"),
                })
                occurrences.append({
                    "occurrence_id": occurrence_id,
                    "start": row.get("start"),
                    "end": row.get("end"),
                    "surface": surface,
                    "aliases": normalized_aliases(row.get("aliases")),
                    "runtime_type": runtime_type,
                    "polarity": row.get("polarity", "unknown"),
                    "detector_provenance": row.get("detector_provenance", {
                        "source": "frozen_arms_migration",
                        "score": row.get("score"),
                    }),
                    "overlap_disposition": row.get("overlap_disposition", "accepted"),
                    "decision_id": decision["decision_id"] if decision is not None else None,
                    "controlled": decision is not None,
                    **({"match": dict(row["match"])}
                       if isinstance(row.get("match"), Mapping) else {}),
                    **({"profile_match": dict(row["profile_match"])}
                       if isinstance(row.get("profile_match"), Mapping) else {}),
                })
                if decision is not None:
                    decision["occurrence_ids"].append(occurrence_id)
            if source_documents is not None and doc_id in source_documents:
                _append_text_anchored_occurrences(
                    source_documents[doc_id], doc_id, decisions_by_key, occurrences)
            occurrences_by_id = {
                str(row["occurrence_id"]): row for row in occurrences
            }
            decisions_by_id = {
                decision["decision_id"]: decision
                for decision in decisions_by_key.values()
            }
            for decision in decisions_by_id.values():
                protected_aliases = []
                for occurrence_id in decision["occurrence_ids"]:
                    for alias in occurrences_by_id[occurrence_id]["aliases"]:
                        if alias not in protected_aliases:
                            protected_aliases.append(alias)
                decision["protected_aliases"] = protected_aliases
                decision["semantic_chain"] = _frozen_semantic_chain(decision, protected_aliases)
            frozen_document = {
                "corpus": corpus,
                "occurrences": occurrences,
                "decisions": list(decisions_by_id.values()),
            }
            frozen_document["environment_document_hash"] = stable_hash(frozen_document)
            documents[doc_id] = frozen_document
    frozen = {"artifact_version": "occurrence-decisions-v1", "documents": documents}
    frozen["environment_hash"] = stable_hash(frozen)
    return frozen


_V2_ACTION_FIELDS = frozenset({
    "fill", "mode", "keep", "source_identity", "aset", "coarseness_rank",
    "level_grounding", "legal", "entails", "forced_placeholder",
})


_V2_OCCURRENCE_FIELDS = frozenset({
    "surface", "type", "runtime_type", "start", "end", "score", "aliases",
    "polarity", "detector_provenance", "overlap_disposition", "match",
    "profile_match", "forced_placeholder",
})


_V2_DETECTOR_ROW_FIELDS = frozenset({
    "text", "surface", "type", "runtime_type", "proposed_runtime_type", "start",
    "end", "score", "source", "raw_label", "recognizer", "status", "reason",
    "min_health_condition_score", "detector_provenance", "overlap_disposition",
})


def _policy_free_action(action: Mapping, runtime_type: str) -> dict:
    clean = {key: action[key] for key in _V2_ACTION_FIELDS if key in action}
    if str(clean.get("mode", "level")) == "placeholder":
        clean["fill"] = None
        clean["placeholder_type"] = runtime_type
    return clean


def _policy_free_ranker_environment(ranker_environment: Mapping) -> dict:
    """Whitelist V2 decision facts from a legacy ranker environment.

    This is deliberately a compatibility adapter, not a policy migration: tau,
    floors, behavior-clone labels, cached risk, and proximity never cross it.
    """
    corpora = {}
    for corpus, documents in (ranker_environment.get("corpora") or {}).items():
        corpora[corpus] = {}
        for doc_id, document in documents.items():
            spans = []
            for span in document.get("spans", []):
                runtime_type = str(span.get("type", span.get("runtime_type", "")))
                spans.append({
                    key: span[key]
                    for key in ("surface", "type", "start", "end", "sent")
                    if key in span
                } | {
                    "type": runtime_type,
                    "actions": [
                        _policy_free_action(action, runtime_type)
                        for action in span.get("actions", [])
                    ],
                })
            corpora[corpus][doc_id] = {"spans": spans}
    return {"corpora": corpora}


def legacy_arms_ranker_environment(arms: Mapping) -> dict:
    """Expose historical action tables through the policy-free V2 whitelist."""
    corpora = {}
    for corpus, documents in arms.items():
        if corpus == "_meta" or not isinstance(documents, Mapping):
            continue
        corpora[corpus] = {}
        for doc_id, document in documents.items():
            table = document.get("v2_action_table", document.get("action_table", {}))
            rows = list(table.values()) if isinstance(table, Mapping) else []
            corpora[corpus][doc_id] = {"spans": rows}
    return _policy_free_ranker_environment({"corpora": corpora})


def _legacy_arms_occurrences(
    arms: Mapping,
    *,
    detector_provenance: Mapping | None = None,
) -> dict[str, list[dict]]:
    """Read only policy-independent rows from a historical arms artifact."""
    result = {}
    for corpus, documents in arms.items():
        if corpus == "_meta" or not isinstance(documents, Mapping):
            continue
        for doc_id, document in documents.items():
            rows = document.get("v2_occurrences")
            if not isinstance(rows, list):
                walk = document.get("tau_walk")
                rows = walk[1] if isinstance(walk, (list, tuple)) and len(walk) > 1 else []
            clean_rows = []
            for row in rows if isinstance(rows, list) else []:
                clean = {key: row[key] for key in _V2_OCCURRENCE_FIELDS if key in row}
                if detector_provenance is not None or row.get("detector_provenance"):
                    clean["detector_provenance"] = {
                        **dict(detector_provenance or {}),
                        **dict(row.get("detector_provenance") or {}),
                        "score": row.get("score"),
                    }
                clean_rows.append(clean)
            result[str(doc_id)] = clean_rows
    return result


def _policy_free_detector_diagnostics(value: object) -> dict:
    if not isinstance(value, Mapping):
        return {}
    clean = {}
    for section in ("accepted", "rejected", "post_detection_rejections"):
        rows = value.get(section)
        if isinstance(rows, list):
            clean[section] = [
                {key: row[key] for key in _V2_DETECTOR_ROW_FIELDS if key in row}
                for row in rows if isinstance(row, Mapping)
            ]
    return clean


def freeze_v2_environment_from_legacy_arms(
    ranker_environment: Mapping,
    arms: Mapping,
    *,
    detector_provenance: Mapping | None = None,
    source_documents: Mapping[str, str] | None = None,
) -> dict:
    """Compatibility boundary from historical arms/env files to the V2 contract.

    The returned representation contains canonical occurrence/decision/action IDs,
    legal lattice actions, typed placeholders, and render offsets. It cannot depend
    on legacy policy outcomes because the adapter uses an explicit field whitelist.
    """
    frozen = freeze_ranker_environment(
        _policy_free_ranker_environment(ranker_environment),
        occurrences_by_document=_legacy_arms_occurrences(
            arms, detector_provenance=detector_provenance,
        ),
        source_documents=source_documents,
    )
    for corpus, documents in arms.items():
        if corpus == "_meta" or not isinstance(documents, Mapping):
            continue
        for doc_id, legacy_document in documents.items():
            document = frozen["documents"].get(str(doc_id))
            if document is None or not isinstance(legacy_document, Mapping):
                continue
            diagnostics = _policy_free_detector_diagnostics(
                legacy_document.get("detector_diagnostics")
            )
            if diagnostics:
                document["detector_diagnostics"] = diagnostics
                document["environment_document_hash"] = stable_hash(document)
    frozen["environment_hash"] = stable_hash({
        key: value for key, value in frozen.items() if key != "environment_hash"
    })
    return frozen


_COUNT_GROUNDING_STATUSES = frozenset({
    "certifying", "model-proposed", "proposal-universe",
})


def _admissible_level_count(row: Mapping, level: str) -> tuple[float | None, dict | None]:
    """Return one row-local admitted count, or an explicit provisional null pair.

    This deliberately does not call ``lookup_count``: that index merges equal level strings
    across profiles and therefore cannot establish the meaning of this decision's action.
    """
    raw_count = (row.get("level_counts") or {}).get(level)
    grounding = (row.get("level_grounding") or {}).get(level)
    if (
        isinstance(raw_count, bool)
        or not isinstance(raw_count, Real)
        or not isfinite(float(raw_count))
        or float(raw_count) < 1.0
        or not isinstance(grounding, Mapping)
    ):
        return None, None
    status = grounding.get("status")
    if status not in _COUNT_GROUNDING_STATUSES or not grounding.get("source_family"):
        return None, None
    if status == "certifying" and not grounding.get("member_set_ref"):
        return None, None
    if status == "model-proposed" and not (
        grounding.get("selector") and grounding.get("count_evidence")
    ):
        return None, None
    if status == "proposal-universe" and not (
        grounding.get("member_set_ref") or grounding.get("generated_universe_ref")
    ):
        return None, None
    return float(raw_count), deepcopy(dict(grounding))


def _matched_profile_entry(
    decision: Mapping,
    occurrences_by_id: Mapping[str, Mapping],
    profile_artifact: Mapping,
) -> tuple[str | None, Mapping | None]:
    """Resolve only the profile identity frozen by the detector-era match.

    Missing, conflicting, or now-absent entries are provisional. Re-running a semantic matcher
    here would change the frozen detection contract and is intentionally forbidden.
    """
    entries = {
        str(profile_match["entry"])
        for occurrence_id in decision.get("occurrence_ids", [])
        if isinstance((occurrence := occurrences_by_id.get(str(occurrence_id))), Mapping)
        and isinstance((profile_match := occurrence.get("profile_match")), Mapping)
        and profile_match.get("entry")
    }
    if len(entries) != 1:
        return None, None
    entry = next(iter(entries))
    runtime_type = str(decision.get("runtime_type", ""))
    row = (profile_artifact.get("profiles") or {}).get(runtime_type, {}).get(entry)
    if not isinstance(row, Mapping):
        return None, None
    return f"{runtime_type}:{entry}", row


def migrate_frozen_environment_count_provenance(
    frozen_environment: Mapping,
    profile_artifact: Mapping,
) -> dict:
    """Enrich frozen action menus with profile-row counts without re-running detection.

    Occurrences and all non-menu decision fields are copied verbatim. Existing action IDs remain
    stable because their mode/fill/legal semantics and legacy ``aset`` fields are unchanged;
    count provenance is reward metadata covered by the menu/document/environment hashes.
    """
    migrated_documents: dict[str, dict] = {}
    for doc_id, source_document in frozen_environment.get("documents", {}).items():
        document = deepcopy(dict(source_document))
        occurrences_by_id = {
            str(row["occurrence_id"]): row for row in source_document.get("occurrences", [])
        }
        migrated_decisions = []
        for source_decision in source_document.get("decisions", []):
            decision = deepcopy(dict(source_decision))
            if decision.get("ranker_selectable", True):
                profile_id, profile_row = _matched_profile_entry(
                    source_decision, occurrences_by_id, profile_artifact,
                )
                decision["profile_id"] = profile_id
                authored_levels = list(profile_row.get("levels", [])) if profile_row else []
                frozen_levels = [
                    str(action.get("fill", ""))
                    for action in source_decision.get("actions", [])
                    if action.get("mode") == "level"
                ]
                merged_authored_levels = [
                    level for level in authored_levels if level in frozen_levels
                ]
                for frozen_index, level in enumerate(frozen_levels):
                    if level in merged_authored_levels:
                        continue
                    next_present = next((
                        candidate for candidate in frozen_levels[frozen_index + 1:]
                        if candidate in merged_authored_levels
                    ), None)
                    if next_present is None:
                        merged_authored_levels.append(level)
                    else:
                        merged_authored_levels.insert(
                            merged_authored_levels.index(next_present), level,
                        )
                migrated_actions = []
                for source_action in source_decision.get("actions", []):
                    action = deepcopy(dict(source_action))
                    mode = action.get("mode")
                    if mode == "level":
                        level = str(action.get("fill", ""))
                        # When the frozen rung no longer exists in the current row, retain its
                        # frozen authored position but make its evidence explicitly provisional.
                        action["authored_level_index"] = merged_authored_levels.index(level)
                        count, grounding = (
                            _admissible_level_count(profile_row, level)
                            if profile_row is not None and level in authored_levels
                            else (None, None)
                        )
                        action["count"] = count
                        action["count_grounding"] = grounding
                    else:
                        for field in ("authored_level_index", "count", "count_grounding"):
                            action.pop(field, None)
                    migrated_actions.append(action)
                decision["actions"] = migrated_actions
                decision["action_menu_hash"] = stable_hash(migrated_actions)
            migrated_decisions.append(decision)
        document["decisions"] = migrated_decisions
        document.pop("environment_document_hash", None)
        document["environment_document_hash"] = stable_hash(document)
        migrated_documents[str(doc_id)] = document
    migrated = {
        "artifact_version": "occurrence-decisions-v2",
        "documents": migrated_documents,
    }
    migrated["environment_hash"] = stable_hash(migrated)
    return migrated


def compare_frozen_environment_semantics(reference: Mapping, candidate: Mapping) -> dict:
    """Compare frozen identity and action semantics, excluding count-only metadata."""
    differences: list[str] = []
    reference_documents = reference.get("documents", {})
    candidate_documents = candidate.get("documents", {})
    if list(reference_documents) != list(candidate_documents):
        differences.append(
            "document_ids: "
            f"reference={list(reference_documents)!r} candidate={list(candidate_documents)!r}"
        )
    for doc_id in reference_documents:
        if doc_id not in candidate_documents:
            continue
        left_document = reference_documents[doc_id]
        right_document = candidate_documents[doc_id]
        left_occurrences = [row.get("occurrence_id") for row in left_document.get("occurrences", [])]
        right_occurrences = [row.get("occurrence_id") for row in right_document.get("occurrences", [])]
        if left_occurrences != right_occurrences:
            differences.append(
                f"{doc_id}:occurrence_ids: reference={left_occurrences!r} "
                f"candidate={right_occurrences!r}"
            )
        left_decisions = left_document.get("decisions", [])
        right_decisions = right_document.get("decisions", [])
        left_ids = [row.get("decision_id") for row in left_decisions]
        right_ids = [row.get("decision_id") for row in right_decisions]
        if left_ids != right_ids:
            differences.append(
                f"{doc_id}:decision_ids: reference={left_ids!r} candidate={right_ids!r}"
            )
        right_by_id = {row.get("decision_id"): row for row in right_decisions}
        for left in left_decisions:
            decision_id = left.get("decision_id")
            right = right_by_id.get(decision_id)
            if right is None:
                continue
            for field in ("action_id", "fill", "mode"):
                left_values = [action.get(field) for action in left.get("actions", [])]
                right_values = [action.get(field) for action in right.get("actions", [])]
                if left_values != right_values:
                    differences.append(
                        f"{doc_id}:{decision_id}:{field}s: reference={left_values!r} "
                        f"candidate={right_values!r}"
                    )
            left_order = [
                action.get("fill") for action in left.get("actions", [])
                if action.get("mode") == "level"
            ]
            right_order = [
                action.get("fill") for action in right.get("actions", [])
                if action.get("mode") == "level"
            ]
            if left_order != right_order:
                differences.append(
                    f"{doc_id}:{decision_id}:authored_order: reference={left_order!r} "
                    f"candidate={right_order!r}"
                )
    return {
        "verdict": "semantic change" if differences else "count-only compatible",
        "differences": differences,
        "reference_document_count": len(reference_documents),
        "candidate_document_count": len(candidate_documents),
    }


def render_frozen_action_vector(
    source: str,
    frozen_document: Mapping,
    action_vector: Mapping[str, str],
) -> tuple[str, list[dict]]:
    """Render one V2 action vector using only frozen offsets and action semantics."""
    decisions = {
        str(decision["decision_id"]): decision
        for decision in frozen_document.get("decisions", [])
    }
    chosen = {}
    placeholder_by_decision = {}
    counters: dict[str, int] = {}
    used_fills = {}
    for decision in frozen_document.get("decisions", []):
        decision_id = str(decision["decision_id"])
        selected_id = str(action_vector[decision_id])
        selected = next(
            (action for action in decision["actions"]
             if str(action["action_id"]) == selected_id),
            None,
        )
        if selected is None or not selected.get("legal", True):
            raise ValueError(f"illegal or unknown action for decision {decision_id}")
        if selected["mode"] == "placeholder":
            runtime_type = str(selected.get("placeholder_type") or decision["runtime_type"])
            token_type = placeholder_type_token(runtime_type)
            counters[token_type] = counters.get(token_type, 0) + 1
            placeholder_by_decision[decision_id] = placeholder_token(
                runtime_type, counters[token_type]
            )
        elif selected["mode"] == "level":
            fill_key = canon(str(selected.get("fill", "")))
            owner = used_fills.setdefault(fill_key, decision_id)
            if owner != decision_id:
                raise ValueError(f"non-injective V2 action vector fill {selected.get('fill')!r}")
        chosen[decision_id] = selected

    replacements = []
    for occurrence in frozen_document.get("occurrences", []):
        decision_id = occurrence.get("decision_id")
        if decision_id is None:
            continue
        decision_id = str(decision_id)
        if decision_id not in decisions or decision_id not in chosen:
            raise ValueError(f"unresolved controlled occurrence decision {decision_id}")
        selected = chosen[decision_id]
        original = source[int(occurrence["start"]):int(occurrence["end"])]
        if selected["mode"] == "keep":
            replacement = original
        elif selected["mode"] == "placeholder":
            replacement = placeholder_by_decision[decision_id]
        else:
            replacement = str(selected["fill"])
            if original[:1].isupper() and replacement:
                replacement = replacement[:1].upper() + replacement[1:]
        replacements.append({
            "occurrence_id": occurrence["occurrence_id"],
            "decision_id": decision_id,
            "start": int(occurrence["start"]),
            "end": int(occurrence["end"]),
            "surface": original,
            "replacement": replacement,
            "action_id": selected["action_id"],
            "mode": selected["mode"],
        })
    ordered = sorted(replacements, key=lambda row: (row["start"], row["end"]))
    for left, right in zip(ordered, ordered[1:]):
        if right["start"] < left["end"]:
            raise ValueError("overlapping frozen V2 occurrences cannot be rendered")
    rendered = source
    for row in reversed(ordered):
        rendered = rendered[:row["start"]] + row["replacement"] + rendered[row["end"]:]
    return rendered, ordered


def frozen_occurrences_from_arms(
    arms: Mapping,
    *,
    detector_provenance: Mapping | None = None,
) -> dict[str, list[dict]]:
    """Legacy compatibility reader; V2 callers use the frozen environment directly."""
    return _legacy_arms_occurrences(arms, detector_provenance=detector_provenance)
