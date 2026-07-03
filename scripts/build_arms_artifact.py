"""Build and persist the constructed-arms artifact: doc_p + R per (corpus, doc, arm).

Detection is nondeterministic across processes on long docs (measured 2026-07-03: 3/6
clinical doc_p hashes differ between fresh runs — borderline GLiNER scores under ROCm
fp16). Recomputing arms per script therefore breaks remote-cache reuse and run-to-run
reproducibility. Fix: build arms ONCE here, persist, and have every consumer (gate,
diagnostics, training env) load the artifact instead of re-detecting.

Run: PYTHONPATH=src:scripts .venv/bin/python -u scripts/build_arms_artifact.py
"""
import json
import sys
import time
from pathlib import Path

from cloak.corpora import load_task_docs
from cloak.detect import Detector

sys.path.append(str(Path(__file__).resolve().parent / "spikes"))
from surrogate_validation import build_arms  # noqa: E402

ARTIFACT = Path("data/task_arms_tau0.02.json")
TAU = 0.02
CORPORA = ("clinical", "enron", "aeslc")
LIMIT = 16


def load_artifact() -> dict:
    """{corpus: {doc_id: {arm: [doc_p, R]}}} — consumers use this, never re-detect."""
    return json.loads(ARTIFACT.read_text())


def main():
    t0 = time.time()
    det = Detector()
    art = {}
    for corpus in CORPORA:
        docs = load_task_docs(corpus, LIMIT)
        art[corpus] = {d["id"]: {arm: [doc_p, R] for arm, (doc_p, R) in
                                 build_arms(d["text"], det.detect(d["text"]), TAU).items()}
                       for d in docs}
        print(f"[{corpus}] {len(docs)} docs {time.time()-t0:.0f}s", flush=True)
    ARTIFACT.write_text(json.dumps(art, indent=1))
    print(f"wall {time.time()-t0:.0f}s -> {ARTIFACT}")


if __name__ == "__main__":
    main()
