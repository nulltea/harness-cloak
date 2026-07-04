"""Annotate the arms artifact with anonymity-set counts and keep-original actions.

In-place, idempotent, NO re-detection (the artifact's spans/lattices/walk decisions are
frozen; spec §3.3-5). Level actions gain "aset"; each span gains a keep-original action
inserted before the trailing placeholder (level indices unchanged; bc_action remapped when
it pointed at the placeholder). Placeholder actions carry no aset — they are always legal.

Run: PYTHONPATH=src .venv/bin/python scripts/annotate_lattice_counts.py
"""
import json
from pathlib import Path

from cloak.anonymity import aset_count

PATH = Path("data/task_arms_tau0.02.json")

art = json.loads(PATH.read_text())
n_spans = n_keep = 0
for corpus, per_doc in art.items():
    for doc_id, entry in per_doc.items():
        for key, span in entry.get("action_table", {}).items():
            acts = span["actions"]
            assert acts[-1]["mode"] == "placeholder", (doc_id, key)
            n_spans += 1
            for a in acts:
                if a["mode"] == "level":
                    a["aset"] = aset_count(a["fill"], span["type"], span["surface"],
                                           strict=True)  # certifying mode, always
            if not any(a.get("keep") for a in acts):  # idempotency
                keep = {"fill": span["surface"], "mode": "level", "keep": True,
                        "walk_risk": 1.0, "p6": 1.0, "aset": 1.0}
                acts.insert(len(acts) - 1, keep)
                if span["bc_action"] == len(acts) - 1 - 1:  # pointed at old placeholder
                    span["bc_action"] = len(acts) - 1
                n_keep += 1
            assert acts[-1]["mode"] == "placeholder"
            bc = acts[span["bc_action"]]
            assert bc["mode"] == "placeholder" or not bc.get("keep"), (doc_id, key)

PATH.write_text(json.dumps(art, indent=1))  # match build_arms_artifact.py format
print(f"annotated {n_spans} spans, inserted {n_keep} keep actions -> {PATH}")
