"""Diagnostic for the stage-1 ranker NULL: how many train probes CAN flip under any
single-action counterfactual, and how many DO.

For each trainable doc: baseline per-probe u_qa F1 at the BC point; then for every probe
whose surface is a multi-legal decision span, swap that one span's action to each legal
alternative (holding the rest at BC) and re-score that probe. A "flip" = |f1_cf - f1_bc|
>= 0.5. Answers whether the utility term is structurally silent (few flippable probes)
vs reader-insensitive (flippable but never flips).

Run: PYTHONPATH=src:scripts .venv/bin/python -u scripts/spikes/probe_flip_scan.py
"""
import json

from build_arms_artifact import load_artifact
from cloak.corpora import load_task_docs
from cloak.train.reward import u_qa

from train_ranker import assemble

env = json.loads(open("data/ranker_env.json").read())
art = load_artifact()
tau = env["tau"]

n_probes = n_flippable = n_flips = 0
f1_hist = {"0": 0, "mid": 0, "1": 0}
flip_examples = []

for corpus, per_doc in env["corpora"].items():
    texts = {d["id"]: d["text"] for d in load_task_docs(corpus, 16)}
    for doc_id, d in per_doc.items():
        if not d["trainable"] or not d["spans"]:
            continue
        spans = []
        for s in d["spans"]:
            s = dict(s)
            s["legal"] = [i for i, a in enumerate(s["actions"])
                          if a["mode"] == "placeholder" or a["walk_risk"] < tau]
            spans.append(s)
        bc = {s["surface"].lower(): s["actions"][s["bc_action"]] for s in spans}
        doc_p, R = assemble(texts[doc_id], d["R_walk"] if "R_walk" in d
                            else art[corpus][doc_id]["tau_walk"][1], spans, bc)
        R_walk = art[corpus][doc_id]["tau_walk"][1]
        probes = d["probes"]["train"]
        _, base_det = u_qa(doc_p, R, probes)
        by_surface = {s["surface"].lower(): s for s in spans}
        for p, bd in zip(probes, base_det):
            n_probes += 1
            f1 = bd["f1"]
            f1_hist["1" if f1 >= 0.99 else "0" if f1 <= 0.01 else "mid"] += 1
            s = by_surface.get(p["surface"].lower())
            if s is None or len(s["legal"]) < 2:
                continue
            n_flippable += 1
            used = {a["fill"].lower() for k, a in bc.items()
                    if a["mode"] == "level" and k != p["surface"].lower()}
            for i in s["legal"]:
                a = s["actions"][i]
                if a is bc[p["surface"].lower()]:
                    continue
                if a["mode"] == "level" and a["fill"].lower() in used:
                    continue  # injectivity-masked in rollouts
                cf = dict(bc)
                cf[p["surface"].lower()] = a
                doc_cf, R_cf = assemble(texts[doc_id], R_walk, spans, cf)
                _, det = u_qa(doc_cf, R_cf, [p])
                if abs(det[0]["f1"] - f1) >= 0.5:
                    n_flips += 1
                    flip_examples.append(
                        {"doc": doc_id, "surface": p["surface"], "bc_f1": f1,
                         "cf_f1": det[0]["f1"],
                         "cf_action": a["fill"] if a["mode"] == "level" else "PLACEHOLDER"})
                    break  # count probes-that-can-flip, not action pairs

print(f"probes={n_probes}  baseline f1 hist: {f1_hist}")
print(f"probes on multi-legal spans (flippable in principle)={n_flippable}")
print(f"probes that ACTUALLY flip under some single-action counterfactual={n_flips}")
for e in flip_examples:
    print("  FLIP:", e)
