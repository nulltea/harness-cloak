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

from cloak.anonymity import aset_count
from cloak.corpora import load_task_docs
from cloak.detect import Detector
from cloak.probe import fill_proximity, walk_risk

sys.path.append(str(Path(__file__).resolve().parent / "spikes"))
from surrogate_validation import build_arms  # noqa: E402

ARTIFACT = Path("data/task_arms_tau0.02.json")
TAU = 0.02
CORPORA = ("clinical", "enron", "aeslc")
LIMIT = 16


def load_artifact(path: str | Path = ARTIFACT) -> dict:
    """{corpus: {doc_id: {arm: [doc_p, R], action_table: {...}}}} — consumers use this,
    never re-detect and never recompute risks (both are build-time-only: detection is
    process-nondeterministic, and walk_risk depends on the distractor-pools snapshot).
    Default is the frozen historical artifact; pass a path for the pilot artifact."""
    return json.loads(Path(path).read_text())


def _sent_around(text: str, start: int, end: int) -> str:
    lo = max(text.rfind(".", 0, start), text.rfind("\n", 0, start)) + 1
    his = [i for i in (text.find(".", end), text.find("\n", end)) if i != -1]
    return text[lo:min(his) + 1 if his else len(text)].strip()


def action_table(text: str, R: list[dict]) -> dict:
    """Per unique quasi span: full action list (levels ∪ placeholder) with walk_risk + p6,
    computed in the SAME process as the walk so the walk's accepted level is consistent
    with its stored risk. Spec §2 Phase 0a."""
    table, seen = {}, set()
    # iterate in the walk's own processing order (right-to-left): the dedup then keeps the
    # exact occurrence whose sentence the walk scored — table risks match walk decisions
    for e in sorted(R, key=lambda e: -e["start"]):
        key = e["surface"].lower()
        if key in seen or not e.get("lattice"):
            continue
        seen.add(key)
        sent = _sent_around(text, e["start"], e["end"])
        actions = []
        for lvl in e["lattice"]:
            sent_f = sent.replace(e["surface"], lvl) if e["surface"] in sent else lvl
            actions.append({"fill": lvl, "mode": "level",
                            "walk_risk": round(walk_risk(sent_f, e["surface"], lvl,
                                                         e["type"]), 4),
                            "p6": round(fill_proximity(lvl, e["surface"]), 4),
                            # anonymity-set count (spec Phase-0 step 2): the k-floor legality
                            # mask reads this; without it every level fails the floor -> desert.
                            "aset": round(aset_count(lvl, e["type"], e["surface"],
                                                     strict=True), 4)})
        actions.append({"fill": None, "mode": "placeholder", "walk_risk": 0.0, "p6": 0.0})
        bc = (len(actions) - 1 if e["action"] == "placeholder" else
              next(i for i, a in enumerate(actions) if a["mode"] == "level"
                   and a["fill"].lower() == e["replacement"].lower()))
        table[key] = {"surface": e["surface"], "type": e["type"], "start": e["start"],
                      "end": e["end"], "sent": sent, "actions": actions, "bc_action": bc}
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-docs", type=int, default=LIMIT,
                    help="docs detected per corpus (pilot scale > 16 needs more docs)")
    ap.add_argument("--corpora", default=",".join(CORPORA),
                    help="comma-separated registered corpus names")
    ap.add_argument("--out", default=str(ARTIFACT),
                    help="output artifact path (default: the frozen historical artifact — "
                         "override for a pilot artifact; NEVER overwrite the frozen one)")
    args = ap.parse_args()
    out = Path(args.out)

    t0 = time.time()
    det = Detector()
    art = {}
    for corpus in args.corpora.split(","):
        docs = load_task_docs(corpus, args.n_docs)
        art[corpus] = {}
        for d in docs:
            arms = build_arms(d["text"], det.detect(d["text"]), TAU)
            entry = {arm: [doc_p, R] for arm, (doc_p, R) in arms.items()}
            entry["action_table"] = action_table(d["text"], arms["tau_walk"][1])
            art[corpus][d["id"]] = entry
        print(f"[{corpus}] {len(docs)} docs {time.time()-t0:.0f}s", flush=True)
    out.write_text(json.dumps(art, indent=1))
    print(f"wall {time.time()-t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
