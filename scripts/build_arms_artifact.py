"""Build and persist the constructed-arms artifact: doc_p + R per (corpus, doc, arm).

Detection is nondeterministic across processes on long docs (measured 2026-07-03: 3/6
clinical doc_p hashes differ between fresh runs — borderline GLiNER scores under ROCm
fp16). Recomputing arms per script therefore breaks remote-cache reuse and run-to-run
reproducibility. Fix: build arms ONCE here, persist, and have every consumer (gate,
diagnostics, training env) load the artifact instead of re-detecting.

Run: PYTHONPATH=src:scripts .venv/bin/python -u scripts/build_arms_artifact.py
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cloak.span_gate as span_gate
from cloak.anonymity import aset_count
from cloak.corpora import load_task_docs
from cloak.detect import Detector, FINE_LABELS, GLINER_LABELS, QA_V2_CLINICAL_LABELS
from cloak.probe import fill_proximity, walk_risk
from cloak.runtime_types import PLACEHOLDER_RE
from cloak.substitute import prepare_spans_for_substitution

sys.path.append(str(Path(__file__).resolve().parent / "spikes"))
from surrogate_validation import build_arms  # noqa: E402

ARTIFACT = Path("data/task_arms_tau0.02.json")
TAU = 0.02
CORPORA = ("clinical", "enron", "aeslc")
LIMIT = 16
QA_V2_CLINICAL_MODEL = "knowledgator/gliner-pii-large-v1.0"
QA_V2_CLINICAL_THRESHOLD = 0.35
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
    controlled_types: frozenset[str] | None = None,
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


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-docs", type=int, default=LIMIT,
                    help="docs detected per corpus (pilot scale > 16 needs more docs)")
    ap.add_argument("--corpora", default=",".join(CORPORA),
                    help="comma-separated registered corpus names")
    ap.add_argument("--out", default=str(ARTIFACT),
                    help="output artifact path (default: the frozen historical artifact — "
                         "override for a pilot artifact; NEVER overwrite the frozen one)")
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


def make_detector(args, profile: str):
    if args.detector_config == "qa-v2-clinical":
        if profile != "clinical":
            raise ValueError(
                f"qa-v2-clinical requires a clinical profile, got {profile!r}"
            )
        return Detector(
            gliner_model=QA_V2_CLINICAL_MODEL,
            threshold=QA_V2_CLINICAL_THRESHOLD,
            profile="clinical",
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
        text, detection.spans, reject_demographic_other=qa_v2
    )
    arms = build_arms(text, spans, tau)
    entry = {arm: [doc_p, records] for arm, (doc_p, records) in arms.items()}
    if qa_v2 and any(
        row.get("type") == "demographic-other"
        for _doc_p, records in entry.values()
        if isinstance(records, list)
        for row in records
    ):
        raise ValueError("qa-v2-clinical cannot freeze demographic-other")
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


def main():
    args = parse_args()
    out = Path(args.out)
    corpora = args.corpora.split(",")

    t0 = time.time()
    qa_v2 = args.detector_config == "qa-v2-clinical"
    detectors: dict[str, Detector] = {}   # one detector per distinct profile (reuses the load)
    art = {}
    for corpus in corpora:
        profile = profile_for(corpus)
        det = detectors.setdefault(profile, None)
        if det is None:
            det = detectors[profile] = make_detector(args, profile)
        docs = load_task_docs(corpus, args.n_docs)
        art[corpus] = {}
        for d in docs:
            if qa_v2:
                detection = det.detect_with_diagnostics(d["text"])
                entry = document_entry_from_detection(d["text"], detection, tau=TAU, qa_v2=True)
            else:
                arms = build_arms(d["text"], det.detect(d["text"]), TAU)
                entry = {arm: [doc_p, R] for arm, (doc_p, R) in arms.items()}
            controlled_types = QA_V2_CONTROLLED_TYPES if qa_v2 else None
            entry["action_table"] = action_table(
                d["text"], entry["tau_walk"][1], controlled_types=controlled_types
            )
            art[corpus][d["id"]] = entry
        print(f"[{corpus}] {len(docs)} docs (profile={profile}) {time.time()-t0:.0f}s", flush=True)
    art["_meta"] = {
        "gate_fingerprint": span_gate.gate_fingerprint(),
        "profiles": {c: profile_for(c) for c in corpora},
        "detector": _detector_metadata(args, corpora),
    }
    out.write_text(json.dumps(art, indent=1))
    print(f"wall {time.time()-t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
