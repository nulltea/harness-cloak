"""Build policy-free Ranker-v2 environments (or explicit legacy-v1 compatibility envs).

Implements spec §2 Phase 0a/0b (docs/specs/RL/surrogate-ranker-infiller.md): per decision
span — the action table (lattice levels ∪ generic placeholder) with precomputed P4 walk_risk
and P6 fill-proximity per action, inert k-floor plumbing, and the behavior-clone label; per
document — QA probes with a persisted, seeded train/held-out split.

Consumes the arms artifact (detection is process-nondeterministic and walk_risk depends on the
pools snapshot; spec §3.3-5) — spans, NLI-gated lattices, per-action risks/proximities, and the
walk's behavior-clone labels all come from the artifact's embedded action tables; this script
never recomputes them. Placeholder actions carry mode only; concrete <TYPE_n> tokens are
assigned at assemble() time.

Output: data/ranker_env.json
Run: PYTHONPATH=src:scripts .venv/bin/python -u scripts/build_ranker_env.py
"""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

from build_arms_artifact import ARTIFACT, CORPORA, load_artifact

from cloak.corpora import load_task_docs
from cloak.runtime_types import RUNTIME_TYPES
from cloak.train.probes import probes_for_docs
from cloak.train.qa_builder import (
    freeze_v2_environment_from_legacy_arms,
    legacy_arms_ranker_environment,
)

OUT = Path("data/ranker_env.json")
TAU = 0.02
HELD_OUT_FRAC = 0.3   # per-doc probe split; n==1 -> train (no held-out; documented)
SPLIT_SEED = 0


def inert_runtime_floors() -> dict[str, float]:
    """Retired runtime legality floors, kept as all-ones plumbing pending grounded counts."""
    return {t: 1.0 for t in RUNTIME_TYPES}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-docs", type=int,
                    help="embedded documents to emit in total; legacy mode defaults to 16")
    ap.add_argument("--corpora", default=",".join(CORPORA),
                    help="comma-separated registered corpus names (e.g. include lexsum)")
    ap.add_argument("--arms", default=str(ARTIFACT),
                    help="input arms artifact path (default: the frozen historical artifact)")
    ap.add_argument("--out", default=str(OUT),
                    help="output env path (default: the frozen env — override for the pilot env)")
    ap.add_argument("--skip-probes", action="store_true",
                    help="build spans/splits without teacher-generated QA probes")
    ap.add_argument("--legacy-v1", action="store_true",
                    help="emit the retired tau/probe/BC compatibility environment for v1 scripts")
    args = ap.parse_args()
    out = Path(args.out)

    t0 = time.time()
    raw_art = json.loads(Path(args.arms).read_text())
    arms_meta = dict(raw_art.pop("_meta", {}) or {})
    art = raw_art
    if not args.legacy_v1:
        requested = args.corpora.split(",")
        embedded_documents = {}
        for corpus in requested:
            for doc_id, entry in art.get(corpus, {}).items():
                if args.n_docs is not None and len(embedded_documents) >= args.n_docs:
                    break
                if isinstance(entry, dict) and isinstance(entry.get("v2_frozen_input"), dict):
                    embedded_documents[doc_id] = entry["v2_frozen_input"]
            if args.n_docs is not None and len(embedded_documents) >= args.n_docs:
                break
        if embedded_documents:
            frozen_version = (
                (arms_meta.get("v2_frozen_environment") or {}).get("artifact_version")
                or "occurrence-decisions-v1"
            )
            frozen = {
                "artifact_version": frozen_version,
                "documents": embedded_documents,
            }
            payload = json.dumps(
                frozen, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            ).encode("utf-8")
            frozen["environment_hash"] = "sha256:" + hashlib.sha256(payload).hexdigest()
        else:
            source_documents = {}
            for corpus in requested:
                source_documents.update({
                    row["id"]: row["text"]
                    for row in load_task_docs(corpus, args.n_docs or 16)
                })
            frozen = freeze_v2_environment_from_legacy_arms(
                legacy_arms_ranker_environment(art), art,
                source_documents=source_documents,
            )
        selected_docs = {
            doc_id: document
            for doc_id, document in frozen["documents"].items()
            if any(doc_id in art.get(corpus, {}) for corpus in requested)
        }
        frozen = {
            "artifact_version": frozen["artifact_version"],
            "documents": selected_docs,
            "environment_hash": frozen["environment_hash"],
        }
        # Recompute via the compatibility freezer after limiting corpora only when
        # building all requested docs; per-doc hashes remain canonical either way.
        env = {
            "artifact_version": (
                "ranker-v2-environment-v2"
                if frozen["artifact_version"] == "occurrence-decisions-v2"
                else "ranker-v2-environment-v1"
            ),
            "compatibility_adapter": (
                "frozen-arms-count-provenance-v1"
                if frozen["artifact_version"] == "occurrence-decisions-v2"
                else "legacy-arms-policy-free-v1"
            ),
            "frozen_environment": frozen,
            "corpora": {
                corpus: {
                    doc_id: {
                        "decisions": document["decisions"],
                        "occurrences": document["occurrences"],
                        "policy_decision_ids": [
                            decision["decision_id"] for decision in document["decisions"]
                            if decision.get("ranker_selectable", True)
                        ],
                        "trainable": any(
                            decision.get("ranker_selectable", True)
                            for decision in document["decisions"]
                        ),
                    }
                    for doc_id, document in selected_docs.items()
                    if doc_id in art.get(corpus, {})
                }
                for corpus in requested
            },
        }
        # The V2 training artifact must be self-describing but never carry legacy
        # policy parameters or behavior-clone/probe provenance.
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(env, indent=1))
        print(f"wall {time.time()-t0:.0f}s -> {out} (ranker-v2, docs={len(selected_docs)})")
        return

    # Explicit legacy adapter below: retained only for v1 scripts.
    env = {"tau": TAU,                    # legacy walk_risk mask — provenance only
           # Floors retired to 1.0 pending grounded counts; downstream plumbing stays inert.
           "k_floors": inert_runtime_floors(),
           "risk_measure": "aset (anonymity-set count); walk_risk retained offline-only",
           "split_seed": SPLIT_SEED, "held_out_frac": HELD_OUT_FRAC,
           "probe_models": {"walk_risk": "EleutherAI/pythia-410m (contrastive re-id)",
                            "p6": "all-MiniLM-L6-v2 cos(fill, original)"},
           "corpora": {}}
    for corpus in args.corpora.split(","):
        docs = {d["id"]: d for d in load_task_docs(corpus, args.n_docs or 16)}
        if args.skip_probes:
            probes = {}
        else:
            probes = probes_for_docs(
                list(docs.values()),
                {i: arms["tau_walk"][1] for i, arms in art[corpus].items()},
                workers=6)
        env["corpora"][corpus] = {}
        for doc_id, arms in art[corpus].items():
            spans = []
            for row in arms["action_table"].values():
                row = dict(row)
                spans.append(row)
            ps = sorted(probes.get(doc_id, []), key=lambda p: p["surface"].lower())
            rng = random.Random(f"{SPLIT_SEED}|{doc_id}")
            n_held = int(len(ps) * HELD_OUT_FRAC) if len(ps) >= 2 else 0
            held_idx = set(rng.sample(range(len(ps)), n_held)) if n_held else set()
            env["corpora"][corpus][doc_id] = {
                "spans": spans,
                "probes": {"train": [p for i, p in enumerate(ps) if i not in held_idx],
                           "held_out": [p for i, p in enumerate(ps) if i in held_idx]},
                # trainable needs BOTH a utility signal and a decision to make
                "trainable": bool(len(ps) - len(held_idx)) and bool(spans),
            }
        n_spans = sum(len(v["spans"]) for v in env["corpora"][corpus].values())
        n_train = sum(v["trainable"] for v in env["corpora"][corpus].values())
        print(f"[{corpus}] decision spans={n_spans} trainable docs={n_train}/{len(docs)} "
              f"{time.time()-t0:.0f}s", flush=True)
    out.write_text(json.dumps(env, indent=1))
    print(f"wall {time.time()-t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
