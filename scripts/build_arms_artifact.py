"""Build and persist the constructed-arms artifact: doc_p + R per (corpus, doc, arm).

Detection is nondeterministic across processes on long docs (measured 2026-07-03: 3/6
clinical doc_p hashes differ between fresh runs — borderline GLiNER scores under ROCm
fp16). Recomputing arms per script therefore breaks remote-cache reuse and run-to-run
reproducibility. Fix: build arms ONCE here, persist, and have every consumer (gate,
diagnostics, training env) load the artifact instead of re-detecting.

Run: PYTHONPATH=src:scripts .venv/bin/python -u scripts/build_arms_artifact.py
"""
import argparse
import hashlib
import json
import sys
import time
from copy import deepcopy
from pathlib import Path

import cloak.detection.span_gate as span_gate
from cloak.lattice.anonymity import aset_count
from cloak.corpora import load_task_docs
from cloak.detection.detect import Detector, FINE_LABELS, GLINER_LABELS, QA_V2_CLINICAL_LABELS
from cloak.lattice.profiles import DEFAULT_PROFILE_PATH, resolve_missing_drug_aliases
from cloak.detection.probe import fill_proximity, walk_risk
from cloak.runtime_types import DIRECT_TYPES, PLACEHOLDER_RE
from cloak.detection.span_prep import freeze_policy_free_candidates, prepare_spans_for_substitution
from cloak.qa.audit import build_environment_audit, write_audit_sidecars
from cloak.qa.freeze import (
    freeze_v2_environment_from_legacy_arms,
    legacy_arms_ranker_environment,
    migrate_frozen_environment_count_provenance,
)

def build_arms(text: str, spans: list, tau: float) -> dict[str, tuple[str, list[dict]]]:
    """Legacy-arms action tables (moved from the retired surrogate_validation spike)."""
    from cloak.detection.span_prep import substitute

    arms = {
        "no_privacy": (text, []),
        "tau_walk": substitute(text, spans, tau=tau),
        "all_floor": substitute(text, spans, tau=-1.0),
    }  # risk never < -1 -> coarsest level
    out, R = text, []
    for s in sorted(spans, key=lambda s: -s.start):
        R.append({
            "surface": s.text,
            "type": s.type,
            "action": "generalize",
            "replacement": "[REDACTED]",
        })
        out = out[: s.start] + "[REDACTED]" + out[s.end :]
    arms["suppression"] = (out, R[::-1])
    return arms


ARTIFACT = Path("data/task_arms_tau0.02.json")
TAU = 0.02
CORPORA = ("clinical", "enron", "aeslc")
LIMIT = 16
QA_V2_CLINICAL_MODEL = "knowledgator/gliner-pii-large-v1.0"
QA_V2_CLINICAL_THRESHOLD = 0.35
# Health-condition admission gate: GLiNER's `condition` label fires at low
# confidence on exam-findings/modifiers (edema/erythema/immunocompromised/acute
# exacerbation ~0.38-0.68) vs real diagnoses (>=0.74); 0.5 drops the findings
# that pollute relation candidates without touching diagnoses. See
# research-wiki/experiments/detector-finding-vs-diagnosis-separation.md.
QA_V2_CLINICAL_MIN_CONDITION_SCORE = 0.5
QA_V2_CLINICAL_LABEL_SCHEMA = "knowledgator-native-clinical-v1"
QA_V2_CONTROLLED_TYPES = frozenset({
    "LOC", "drug", "health-condition", "medical-procedure",
})

# Detector profile per corpus (spec hard rule: train/deploy consistency). Clinical text must
# go through the production "clinical" profile so RL artifacts see the same span gate as
# deployment; the other corpora keep the historical "reddit" default.
DEFAULT_PROFILE = "reddit"
CORPUS_PROFILES = {"clinical": "clinical", "aci": "clinical", "mts": "clinical"}


def profile_for(corpus: str) -> str:
    return CORPUS_PROFILES.get(corpus, DEFAULT_PROFILE)


def load_artifact(path: str | Path = ARTIFACT) -> dict:
    """{corpus: {doc_id: {arm: [doc_p, R], action_table: {...}}}} — consumers use this,
    never re-detect and never recompute risks (both are build-time-only: detection is
    process-nondeterministic, and walk_risk depends on the distractor-pools snapshot).
    Default is the frozen historical artifact; pass a path for the pilot artifact.
    The top-level "_meta" provenance block (gate_fingerprint, per-corpus profiles) is
    stripped so callers keep iterating {corpus: {doc_id: ...}} unchanged."""
    art = json.loads(Path(path).read_text())
    art.pop("_meta", None)
    return art


def _sent_around(text: str, start: int, end: int) -> str:
    lo = max(text.rfind(".", 0, start), text.rfind("\n", 0, start)) + 1
    his = [i for i in (text.find(".", end), text.find("\n", end)) if i != -1]
    return text[lo:min(his) + 1 if his else len(text)].strip()


def action_table(
    text: str,
    R: list[dict],
    *,
    controlled_types: set[str] | frozenset[str] | None = None,
) -> dict:
    """Per unique quasi span: full action list (levels ∪ placeholder) with walk_risk + p6,
    computed in the SAME process as the walk so the walk's accepted level is consistent
    with its stored risk. Spec §2 Phase 0a."""
    table, seen = {}, set()
    # iterate in the walk's own processing order (right-to-left): the dedup then keeps the
    # exact occurrence whose sentence the walk scored — table risks match walk decisions
    for e in sorted(R, key=lambda e: -e["start"]):
        key = e["surface"].lower()
        if (
            key in seen
            or not e.get("lattice")
            or (controlled_types is not None and e["type"] not in controlled_types)
        ):
            continue
        sent = _sent_around(text, e["start"], e["end"])
        actions = []
        for lvl in e["lattice"]:
            if PLACEHOLDER_RE.fullmatch(lvl):
                continue
            sent_f = sent.replace(e["surface"], lvl) if e["surface"] in sent else lvl
            actions.append({"fill": lvl, "mode": "level",
                            "walk_risk": round(walk_risk(sent_f, e["surface"], lvl,
                                                         e["type"]), 4),
                            "p6": round(fill_proximity(lvl, e["surface"]), 4),
                            # anonymity-set count: retained as a policy feature/diagnostic.
                            # Runtime k-floor legality was retired to inert env plumbing.
                            "aset": round(aset_count(lvl, e["type"], e["surface"],
                                                     strict=True), 4)})
        if not actions and controlled_types is not None:
            continue
        seen.add(key)
        actions.append({"fill": None, "mode": "placeholder", "walk_risk": 0.0, "p6": 0.0})
        bc = (len(actions) - 1 if e["action"] == "placeholder" else
              next(i for i, a in enumerate(actions) if a["mode"] == "level"
                   and a["fill"].lower() == e["replacement"].lower()))
        table[key] = {"surface": e["surface"], "type": e["type"], "start": e["start"],
                      "end": e["end"], "sent": sent, "actions": actions, "bc_action": bc}
    return table


def v2_action_table(
    text: str,
    records: list[dict],
    *,
    controlled_types: set[str] | frozenset[str],
) -> dict:
    """Policy-free QA/Ranker-v2 menus from a historical frozen record.

    The compatibility read is limited to identity, offsets, lattice levels, and
    profile counts. It never observes the legacy selected action, tau outcome,
    risk, exhaustion, proximity, floors, or behavior-clone label.
    """
    table, seen = {}, set()
    for row in sorted(records, key=lambda value: int(value["start"])):
        key = (str(row.get("type", "")), str(row.get("surface", "")).casefold())
        if (key in seen or row.get("uncontrolled") or
                (row.get("type") not in controlled_types and row.get("type") not in DIRECT_TYPES)):
            continue
        levels = [
            level for level in row.get("lattice", [])
            if isinstance(level, str) and not PLACEHOLDER_RE.fullmatch(level)
        ]
        if not levels and row.get("type") not in DIRECT_TYPES:
            continue
        seen.add(key)
        runtime_type = str(row["type"])
        actions = [
            {
                "fill": level,
                "mode": "level",
                "aset": round(aset_count(
                    level, runtime_type, str(row["surface"]), strict=True,
                ), 4),
                "legal": True,
            }
            for level in levels
        ]
        placeholder_action = {
            "fill": None,
            "mode": "placeholder",
            "placeholder_type": runtime_type,
            "legal": True,
        }
        if runtime_type in DIRECT_TYPES:
            placeholder_action["forced_placeholder"] = True
        actions.append(placeholder_action)
        table["|".join(key)] = {
            "surface": row["surface"], "type": runtime_type,
            "start": row["start"], "end": row["end"],
            "sent": _sent_around(text, row["start"], row["end"]),
            "actions": actions,
        }
    return table


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-docs", type=int,
                    help="documents to migrate in total; detection mode defaults to 16 per corpus")
    ap.add_argument("--corpora", default=",".join(CORPORA),
                    help="comma-separated registered corpus names")
    ap.add_argument("--out", default=str(ARTIFACT),
                    help="output artifact path (default: the frozen historical artifact — "
                         "override for a pilot artifact; NEVER overwrite the frozen one)")
    ap.add_argument("--from-arms",
                    help="migrate embedded v2_frozen_input without detection or corpus reload")
    ap.add_argument("--profiles", default=str(DEFAULT_PROFILE_PATH),
                    help="read-only profile artifact used for row-local count provenance")
    ap.add_argument("--profile-mutation-queue",
                    help="hard-error queue for a detected mutation of --profiles")
    ap.add_argument("--detector-model",
                    help="GLiNER checkpoint/path for detection; omit for Detector's default")
    ap.add_argument("--threshold", type=float,
                    help="detector threshold; omit for Detector's default")
    ap.add_argument("--fine-dem", action="store_true",
                    help="emit fine demographic runtime types instead of coarse DEM")
    ap.add_argument(
        "--detector-config",
        choices=("deployment", "qa-v2-clinical"),
        default="deployment",
        help="pinned detector preset (QA-v2 forbids model/threshold/schema overrides)",
    )
    args = ap.parse_args(argv)
    corpora = [corpus.strip() for corpus in args.corpora.split(",")]
    if any(not corpus for corpus in corpora):
        ap.error("--corpora must contain non-empty comma-separated corpus names")
    args.corpora = ",".join(corpora)
    if args.detector_config == "qa-v2-clinical":
        conflicts = []
        if args.detector_model:
            conflicts.append("--detector-model")
        if args.threshold is not None:
            conflicts.append("--threshold")
        if args.fine_dem:
            conflicts.append("--fine-dem")
        if conflicts:
            ap.error(
                "--detector-config qa-v2-clinical cannot be combined with "
                + ", ".join(conflicts)
            )
        nonclinical = [corpus for corpus in corpora if profile_for(corpus) != "clinical"]
        if nonclinical:
            ap.error(
                "--detector-config qa-v2-clinical requires clinical corpora; got "
                + ", ".join(nonclinical)
            )
    return args


def migrate_arms_artifact(
    source_artifact: dict,
    profile_artifact: dict,
    *,
    n_docs: int | None = None,
) -> dict:
    """Migrate embedded frozen documents while preserving detector-era records verbatim."""
    selected: list[tuple[str, str, dict]] = []
    for corpus, documents in source_artifact.items():
        if corpus == "_meta" or not isinstance(documents, dict):
            continue
        for doc_id, entry in documents.items():
            if n_docs is not None and len(selected) >= n_docs:
                break
            if not isinstance(entry, dict) or not isinstance(entry.get("v2_frozen_input"), dict):
                raise ValueError(f"{corpus}:{doc_id} lacks embedded v2_frozen_input")
            selected.append((corpus, doc_id, entry))
        if n_docs is not None and len(selected) >= n_docs:
            break
    if not selected:
        raise ValueError("--from-arms selected no embedded frozen documents")
    frozen = {
        "artifact_version": "occurrence-decisions-v1",
        "documents": {
            doc_id: deepcopy(entry["v2_frozen_input"])
            for _corpus, doc_id, entry in selected
        },
    }
    migrated_frozen = migrate_frozen_environment_count_provenance(
        frozen, profile_artifact,
    )
    migrated = {"_meta": deepcopy(source_artifact.get("_meta", {}))}
    source_audit = migrated["_meta"].pop("environment_audit", None)
    if source_audit is not None:
        migrated["_meta"]["source_environment_audit"] = source_audit
    migrated["_meta"]["v2_frozen_environment"] = {
        "artifact_version": migrated_frozen["artifact_version"],
        "environment_hash": migrated_frozen["environment_hash"],
        "count_sourcing": "matched-profile-row-level-counts-v1",
    }
    for corpus, doc_id, entry in selected:
        migrated.setdefault(corpus, {})[doc_id] = deepcopy(entry)
        migrated[corpus][doc_id]["v2_frozen_input"] = migrated_frozen["documents"][doc_id]
    migrated["_meta"]["count_migration_audit"] = _count_migration_audit(migrated_frozen)
    return migrated


def _count_migration_audit(frozen_environment: dict) -> dict:
    nulls_by_type: dict[str, int] = {}
    unmatched_by_type: dict[str, int] = {}
    level_actions = 0
    selectable_decisions = 0
    missing_policy_mappings = 0
    nonmonotone = 0
    for document in frozen_environment["documents"].values():
        occurrences = {
            str(row["occurrence_id"]): row for row in document.get("occurrences", [])
        }
        for decision in document.get("decisions", []):
            if not decision.get("ranker_selectable", True):
                continue
            selectable_decisions += 1
            runtime_type = str(decision.get("runtime_type", ""))
            if "profile_id" not in decision:
                missing_policy_mappings += 1
            elif decision["profile_id"] is None:
                unmatched_by_type[runtime_type] = unmatched_by_type.get(runtime_type, 0) + 1
            for occurrence_id in decision.get("occurrence_ids", []):
                occurrence = occurrences.get(str(occurrence_id))
                if occurrence is None or occurrence.get("decision_id") != decision.get("decision_id"):
                    missing_policy_mappings += 1
            levels = [
                action for action in decision.get("actions", [])
                if action.get("mode") == "level" and action.get("legal", True)
            ]
            level_actions += len(levels)
            for action in levels:
                if action.get("count") is None:
                    nulls_by_type[runtime_type] = nulls_by_type.get(runtime_type, 0) + 1
            admitted = [
                action for action in sorted(
                    levels, key=lambda row: int(row["authored_level_index"]),
                )
                if action.get("count") is not None
            ]
            if any(
                float(right["count"]) < float(left["count"])
                for left, right in zip(admitted, admitted[1:])
            ):
                nonmonotone += 1
    audit = {
        "version": "frozen-count-migration-audit-v1",
        "documents": len(frozen_environment["documents"]),
        "decisions": sum(
            len(document.get("decisions", []))
            for document in frozen_environment["documents"].values()
        ),
        "ranker_selectable_decisions": selectable_decisions,
        "level_actions": level_actions,
        "missing_policy_mappings": missing_policy_mappings,
        "nonmonotone_non_null": nonmonotone,
        "provisional_null_level_actions_by_type": dict(sorted(nulls_by_type.items())),
        "unmatched_profile_decisions_by_type": dict(sorted(unmatched_by_type.items())),
    }
    payload = json.dumps(audit, sort_keys=True, separators=(",", ":")).encode("utf-8")
    audit["audit_hash"] = "sha256:" + hashlib.sha256(payload).hexdigest()
    return audit


def make_detector(args, profile: str):
    if args.detector_config == "qa-v2-clinical":
        if profile != "clinical":
            raise ValueError(
                f"qa-v2-clinical requires a clinical profile, got {profile!r}"
            )
        if args.detector_model or args.threshold is not None or args.fine_dem:
            raise ValueError(
                "qa-v2-clinical detector config cannot be combined with detector overrides"
            )
        return Detector(
            gliner_model=QA_V2_CLINICAL_MODEL,
            threshold=QA_V2_CLINICAL_THRESHOLD,
            profile=profile,
            label2type=QA_V2_CLINICAL_LABELS,
        )
    kwargs = {"fine_dem": args.fine_dem, "profile": profile}
    if args.detector_model:
        kwargs["gliner_model"] = args.detector_model
    if args.threshold is not None:
        kwargs["threshold"] = args.threshold
    return Detector(**kwargs)


def document_entry_from_detection(text, detection, *, tau: float, qa_v2: bool) -> dict:
    spans, post_rejections = prepare_spans_for_substitution(
        text, detection.spans, reject_demographic_other=qa_v2,
        min_health_condition_score=QA_V2_CLINICAL_MIN_CONDITION_SCORE if qa_v2 else None,
    )
    if qa_v2:
        records = freeze_policy_free_candidates(text, spans)
        if any(row.get("type") == "demographic-other" for row in records):
            raise ValueError("qa-v2-clinical cannot freeze demographic-other")
        entry = {"v2_occurrences": records}
    else:
        arms = build_arms(text, spans, tau)
        entry = {arm: [doc_p, records] for arm, (doc_p, records) in arms.items()}
    diagnostic = detection.as_dict()
    entry["detector_diagnostics"] = {
        **diagnostic,
        "post_detection_rejections": post_rejections,
    }
    return entry


def _detector_metadata(args, corpora: list[str]) -> dict:
    profiles = {corpus: profile_for(corpus) for corpus in corpora}
    if args.detector_config == "qa-v2-clinical":
        return {
            "config": "qa-v2-clinical",
            "model": QA_V2_CLINICAL_MODEL,
            "threshold": QA_V2_CLINICAL_THRESHOLD,
            "min_health_condition_score": QA_V2_CLINICAL_MIN_CONDITION_SCORE,
            "label_schema": QA_V2_CLINICAL_LABEL_SCHEMA,
            "label_map": dict(QA_V2_CLINICAL_LABELS),
            "controlled_runtime_types": sorted(QA_V2_CONTROLLED_TYPES),
            "presidio": True,
            "profiles": profiles,
        }
    return {
        "config": "deployment",
        "model": args.detector_model or "data/models/pii_gliner_multidomain/checkpoint-2479",
        "threshold": args.threshold if args.threshold is not None else 0.3,
        "label_schema": "fine-dem" if args.fine_dem else "tab-8",
        "label_map": dict(FINE_LABELS if args.fine_dem else GLINER_LABELS),
        "controlled_runtime_types": None,
        "presidio": True,
        "profiles": profiles,
    }


def detector_manifest(args, corpora: list[str]) -> dict:
    """Public detector provenance manifest retained for artifact consumers."""
    return _detector_metadata(args, corpora)


def main():
    args = parse_args()
    out = Path(args.out)
    corpora = args.corpora.split(",")

    t0 = time.time()
    if args.from_arms:
        profile_path = Path(args.profiles)
        profile_bytes = profile_path.read_bytes()
        profile_hash = hashlib.sha256(profile_bytes).hexdigest()
        profile_artifact = json.loads(profile_bytes)
        source_artifact = json.loads(Path(args.from_arms).read_text())
        art = migrate_arms_artifact(
            source_artifact, profile_artifact, n_docs=args.n_docs,
        )
        after_hash = hashlib.sha256(profile_path.read_bytes()).hexdigest()
        if after_hash != profile_hash:
            queue_path = Path(
                args.profile_mutation_queue
                or out.with_name(f"{out.stem}.profile-mutation-queue.jsonl")
            )
            queue_path.parent.mkdir(parents=True, exist_ok=True)
            queue_entry = {
                "kind": "canonical-profile-mutation",
                "profile_path": str(profile_path),
                "before_sha256": profile_hash,
                "after_sha256": after_hash,
                "source_arms": args.from_arms,
                "status": "hard-error",
            }
            queue_path.write_text(json.dumps(queue_entry, sort_keys=True) + "\n")
            raise RuntimeError(
                f"profile artifact changed during migration; queued hard error at {queue_path}"
            )
        art["_meta"]["count_migration"] = {
            "source_arms": args.from_arms,
            "profile_path": str(profile_path),
            "profile_sha256": profile_hash,
            "detector_rerun": False,
            "external_model_calls": 0,
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(art, indent=1))
        document_count = sum(
            len(documents) for corpus, documents in art.items() if corpus != "_meta"
        )
        print(
            f"wall {time.time()-t0:.3f}s -> {out}; "
            f"migrated_docs={document_count}; detector_rerun=false; model_calls=0",
            flush=True,
        )
        return

    qa_v2 = args.detector_config == "qa-v2-clinical"
    detectors: dict[str, Detector] = {}   # one detector per distinct profile (reuses the load)
    art = {}
    source_documents: dict[str, str] = {}
    for corpus in corpora:
        profile = profile_for(corpus)
        det = detectors.setdefault(profile, None)
        if det is None:
            det = detectors[profile] = make_detector(args, profile)
        docs = load_task_docs(corpus, args.n_docs if args.n_docs is not None else LIMIT)
        art[corpus] = {}
        for d in docs:
            source_documents[d["id"]] = d["text"]
            if qa_v2:
                detection = det.detect_with_diagnostics(d["text"])
                # Safe auto data-fix BEFORE the lattice chain bakes: alias an unprofiled brand
                # drug onto its existing generic profile entry (openFDA NDC), persisting to
                # lattice_profiles.json. Strict (no invented entries/levels, no ambiguous brand),
                # so the bake below resolves e.g. a brand to its generic's levels instead of
                # leaving it unprofiled. Clears the profile cache so this doc's bake sees the fix.
                fixed = resolve_missing_drug_aliases(
                    {span.text for span in detection.spans if span.type == "drug"}
                )
                if fixed:
                    print(f"  [{d['id']}] resolved drug aliases -> {fixed}", flush=True)
                entry = document_entry_from_detection(d["text"], detection, tau=TAU, qa_v2=True)
            else:
                arms = build_arms(d["text"], det.detect(d["text"]), TAU)
                entry = {arm: [doc_p, R] for arm, (doc_p, R) in arms.items()}
            controlled_types = QA_V2_CONTROLLED_TYPES if qa_v2 else None
            if qa_v2:
                entry["v2_action_table"] = v2_action_table(
                    d["text"], entry["v2_occurrences"],
                    controlled_types=QA_V2_CONTROLLED_TYPES,
                )
            else:
                entry["action_table"] = action_table(
                    d["text"], entry["tau_walk"][1], controlled_types=controlled_types
                )
            art[corpus][d["id"]] = entry
        print(f"[{corpus}] {len(docs)} docs (profile={profile}) {time.time()-t0:.0f}s", flush=True)
    art["_meta"] = {
        "gate_fingerprint": span_gate.gate_fingerprint(),
        "profiles": {c: profile_for(c) for c in corpora},
        "detector": detector_manifest(args, corpora),
    }
    if qa_v2:
        frozen_v2 = freeze_v2_environment_from_legacy_arms(
            legacy_arms_ranker_environment(art), art,
            detector_provenance=art["_meta"]["detector"],
            source_documents=source_documents,
        )
        for corpus in corpora:
            for doc_id, entry in art[corpus].items():
                entry["v2_frozen_input"] = frozen_v2["documents"][doc_id]
        art["_meta"]["v2_frozen_environment"] = {
            "artifact_version": frozen_v2["artifact_version"],
            "environment_hash": frozen_v2["environment_hash"],
        }
        environment_audit = build_environment_audit(
            frozen_v2, source_documents=source_documents,
        )
        # The QA-v2 artifact is the frozen contract itself.  Do not serialize the
        # temporary legacy arms used by the migration adapter into a V2 output.
        for corpus in corpora:
            for doc_id, entry in list(art[corpus].items()):
                art[corpus][doc_id] = {
                    "v2_frozen_input": entry["v2_frozen_input"],
                }
    else:
        from cloak.qa.audit import build_legacy_environment_audit
        environment_audit = build_legacy_environment_audit(art)
    art["_meta"]["environment_audit"] = {
        "version": environment_audit["version"],
        "audit_hash": environment_audit["audit_hash"],
        "summary_by_code": environment_audit["summary_by_code"],
        "summary_by_action": environment_audit["summary_by_action"],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(art, indent=1))
    audit_paths = write_audit_sidecars(
        environment_audit, out.with_name(f"{out.stem}.environment-audit"),
    )
    print(f"wall {time.time()-t0:.0f}s -> {out}; environment_audit={','.join(map(str, audit_paths))}")


if __name__ == "__main__":
    main()
