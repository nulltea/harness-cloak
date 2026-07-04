"""Phase-0 builder: the stage-1 ranker training environment artifact.

Implements spec §2 Phase 0a/0b (docs/specs/RL/surrogate-ranker-infiller.md): per decision
span — the action table (lattice levels ∪ generic placeholder) with precomputed P4 walk_risk
and P6 fill-proximity per action, the τ-legal mask, and the behavior-clone label (the τ-walk's
own choice); per document — QA probes with a persisted, seeded train/held-out split.

Consumes the arms artifact (detection is process-nondeterministic and walk_risk depends on the
pools snapshot; spec §3.3-5) — spans, NLI-gated lattices, per-action risks/proximities, and the
walk's behavior-clone labels all come from the artifact's embedded action tables; this script
never recomputes them. Placeholder actions carry mode only; concrete <TYPE_n> tokens are
assigned at assemble() time.

Output: data/ranker_env.json
Run: PYTHONPATH=src:scripts .venv/bin/python -u scripts/build_ranker_env.py
"""
import json
import random
import time
from pathlib import Path

from build_arms_artifact import load_artifact

from cloak.anonymity import K_FLOORS
from cloak.corpora import load_task_docs
from cloak.train.probes import probes_for_docs

OUT = Path("data/ranker_env.json")
TAU = 0.02
HELD_OUT_FRAC = 0.3   # per-doc probe split; n==1 -> train (no held-out; documented)
SPLIT_SEED = 0


def main():
    t0 = time.time()
    art = load_artifact()
    env = {"tau": TAU,                    # legacy walk_risk mask — provenance only
           "k_floors": K_FLOORS,          # per-type anonymity-set count floors (the knob)
           "risk_measure": "aset (anonymity-set count); walk_risk retained offline-only",
           "split_seed": SPLIT_SEED, "held_out_frac": HELD_OUT_FRAC,
           "probe_models": {"walk_risk": "EleutherAI/pythia-410m (contrastive re-id)",
                            "p6": "all-MiniLM-L6-v2 cos(fill, original)"},
           "corpora": {}}
    for corpus in ("clinical", "enron", "aeslc"):
        docs = {d["id"]: d for d in load_task_docs(corpus, 16)}
        probes = probes_for_docs(list(docs.values()),
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
        print(f"[{corpus}] decision spans={n_spans} trainable docs={n_train}/16 "
              f"{time.time()-t0:.0f}s", flush=True)
    OUT.write_text(json.dumps(env, indent=1))
    print(f"wall {time.time()-t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
