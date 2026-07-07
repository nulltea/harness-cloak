"""Context-injection ablation — label builder (arm-independent, built ONCE).

Spec: research-wiki/experiments/context-injection-surface-ablation.md.

For each probe-bearing decision span: hold every OTHER span at the floor-walk baseline, force
THIS span to each of its legal actions, and read the span's OWN-PROBE recall off the
(deterministic, cached) round-trip reward. The per-span action->recall profile is the label;
the oracle action is its argmax.

Collision handling MATCHES train time (Finding 3): the counterfactual choice is resolved with
the SAME walk-order rule as floor_walk_choice — earlier spans keep their fills, and a LATER span
colliding with the forced span's fill is downgraded to placeholder (not dropped). The forced
action is skipped only when an EARLIER span already claimed its fill (genuinely illegal), instead
of the old try/except-skip that dropped reachable early-span actions.

Feature row is the POST-REMOVAL set (no walk_risk, no corpus): [is_placeholder, p6,
level_index/4, n_levels/4, log10_aset/9, log10_floor/9] + type one-hot(7) = 13 dims. The ablation
adds a context vector ON TOP of these.

Also emits a per-doc ASSEMBLY BUNDLE (text, R_walk, raw_spans) so the harness's Level 2 can
greedily assemble full-doc choices and score realized joint R_rt.

Run: INFERDPT_LLM_CACHE=data/llm_cache PYTHONPATH=src:scripts \
       .venv/bin/python -u scripts/spikes/context_ablation_labels.py \
       --env data/ranker_env_full.json --arms data/task_arms_full.json \
       --probes data/probes_validated.json --n-docs 60 --workers 6
"""
import argparse
import json
import math
import random
from pathlib import Path

OUT = Path("results/context_ablation_labels.json")


def feats13(span: dict, action_idx: int, floor: float) -> list[float]:
    """Post-removal engineered per-action features (no walk_risk, no corpus)."""
    from cloak.train.ranker import TYPES
    a = span["actions"][action_idx]
    n_lvl = sum(x["mode"] == "level" for x in span["actions"])
    t_oh = [0.0] * len(TYPES)
    t_oh[TYPES.index(span["type"]) if span["type"] in TYPES else TYPES.index("OTHER")] = 1.0
    return [1.0 if a["mode"] == "placeholder" else 0.0,
            a["p6"], min(action_idx, 4) / 4.0, min(n_lvl, 4) / 4.0,
            math.log10(max(a.get("aset", 1e9), 1.0)) / 9.0,
            math.log10(max(floor, 1.0)) / 9.0] + t_oh


def forced_walk_choice(spans: list[dict], forced_key: str, forced_idx: int):
    """floor_walk_choice with span `forced_key` pinned to action `forced_idx`.

    Same walk-order collision rule as scripts/train_ranker.py floor_walk_choice: iterate spans
    in order, keep a `used` set of claimed level fills, downgrade a colliding non-forced span to
    placeholder. Returns None if the forced action's fill was already claimed by an EARLIER span
    (genuinely illegal), else a {surface.lower(): action} choice injective by construction."""
    used, choice = set(), {}
    for s in spans:
        skey = s["surface"].lower()
        if skey == forced_key:
            a = s["actions"][forced_idx]
            if a["mode"] == "level" and a["fill"].lower() in used:
                return None                                  # earlier span took this fill
        else:
            a = s["actions"][s["bc_action"]]
            if a["mode"] == "level" and a["fill"].lower() in used:
                a = s["actions"][next(i for i, x in enumerate(s["actions"])
                                      if x["mode"] == "placeholder")]
        if a["mode"] == "level":
            used.add(a["fill"].lower())
        choice[skey] = a
    return choice


def main():
    from build_arms_artifact import load_artifact
    from train_ranker import assemble, derive_spans, floor_walk_choice

    from cloak.corpora import load_task_docs
    from cloak.train.ranker import span_context
    from cloak.train.reward import canon
    from cloak.train.roundtrip import RT_BASE_URL, RT_MODEL, roundtrip_batch

    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="data/ranker_env_full.json")
    ap.add_argument("--arms", default="data/task_arms_full.json")
    ap.add_argument("--probes", default="data/probes_validated.json")
    ap.add_argument("--n-docs", type=int, default=60,
                    help="docs loaded per corpus; docs beyond the frozen arms artifact are skipped")
    ap.add_argument("--max-spans", type=int, default=0,
                    help="cap on labelled spans (0 = all probe-bearing spans)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    art = load_artifact(args.arms)
    env = json.loads(Path(args.env).read_text())
    probes_art = json.loads(Path(args.probes).read_text())
    probes_all = probes_art["docs"]
    floors = dict(env["k_floors"])

    # floor-walk baseline per doc (same walk-order collision rule as the trainer)
    docs = []
    for corpus, per_doc in env["corpora"].items():
        texts = {d["id"]: d["text"] for d in load_task_docs(corpus, args.n_docs)}
        for doc_id, d in per_doc.items():
            probes = probes_all.get(doc_id, {}).get("train", [])
            if doc_id not in texts or not d.get("trainable") or not d["spans"] or len(probes) < 3:
                continue
            spans, _ = derive_spans(d["spans"], floors, corpus, "cpu")
            docs.append({"id": doc_id, "corpus": corpus, "text": texts[doc_id],
                         "R_walk": art[corpus][doc_id]["tau_walk"][1],
                         "spans": spans, "raw_spans": d["spans"],
                         "floor_choice": floor_walk_choice(spans),   # realized baseline (collision-resolved)
                         "probes": probes})

    # enumerate (probe-bearing span, legal action); others held at floor-walk with collision replay
    jobs, meta = [], []
    labelled = 0
    for d in docs:
        for s in d["spans"]:
            surf = s["surface"]
            own = [j for j, p in enumerate(d["probes"]) if canon(p["surface"]) == canon(surf)]
            if not own:
                continue                        # only spans with a measurable utility signal
            if args.max_spans and labelled >= args.max_spans:
                break
            labelled += 1
            floor_val = floors.get(s["type"], floors.get("OTHER", 100.0))
            ctx_text = span_context(d["text"], s["start"])
            # REALIZED floor-walk baseline (collision-resolved), not bc_action: a collided span
            # is downgraded to placeholder, so bc_action may not be the action actually taken.
            base_action = d["floor_choice"][surf.lower()]
            base_idx = next(k for k, a in enumerate(s["actions"]) if a is base_action)
            for i in s["legal"]:
                choice = forced_walk_choice(d["spans"], surf.lower(), i)
                if choice is None:              # forced fill claimed by an earlier span
                    continue
                try:
                    doc_p, R = assemble(d["text"], d["R_walk"], d["raw_spans"], choice)
                except AssertionError:          # defensive: should not fire post-replay
                    continue
                jobs.append({"corpus": d["corpus"], "doc_p": doc_p, "R": R, "probes": d["probes"]})
                a = s["actions"][i]
                meta.append({"doc_id": d["id"], "corpus": d["corpus"], "surface": surf,
                             "ctx_text": ctx_text, "own": own,
                             "action": {"action_idx": i, "mode": a["mode"], "fill": a.get("fill"),
                                        "feats": feats13(s, i, floor_val),
                                        "is_baseline": i == base_idx}})

    print(f"labelling {labelled} spans -> {len(jobs)} round trips ({len(docs)} docs)", flush=True)
    outs = roundtrip_batch(jobs, workers=args.workers)

    spans_out: dict[tuple, dict] = {}
    for m, o in zip(meta, outs):
        own_recall = max((o["f1s"][j] for j in m["own"]), default=0.0)
        key = (m["doc_id"], m["surface"])
        sp = spans_out.setdefault(key, {"doc_id": m["doc_id"], "corpus": m["corpus"],
                                        "surface": m["surface"], "ctx_text": m["ctx_text"],
                                        "actions": []})
        act = dict(m["action"]); act["own_recall"] = round(own_recall, 4)
        sp["actions"].append(act)

    spans = [s for s in spans_out.values() if len(s["actions"]) >= 2]  # need a choice to judge
    for s in spans:                              # oracle + flat flag per span (harness convenience)
        rec = [a["own_recall"] for a in s["actions"]]
        s["flat"] = max(rec) == min(rec)
        n_base = sum(a["is_baseline"] for a in s["actions"])
        assert n_base == 1, (s["doc_id"], s["surface"], f"{n_base} baseline actions")  # exactly one realized baseline

    # Level-2 assembly bundle: everything the harness needs to greedily assemble + score joint R_rt.
    kept_docs = {(s["doc_id"]) for s in spans}
    bundles = {d["id"]: {"corpus": d["corpus"], "text": d["text"], "R_walk": d["R_walk"],
                         "raw_spans": d["raw_spans"], "probes": d["probes"],
                         "spans": [{"surface": sp["surface"], "type": sp["type"],
                                    "bc_action": sp["bc_action"], "legal": sp["legal"],
                                    "actions": sp["actions"]}
                                   for sp in d["spans"]]}
               for d in docs if d["id"] in kept_docs}

    out = {"meta": {"rt_model": RT_MODEL, "rt_base_url": RT_BASE_URL,
                    "env": args.env, "arms": args.arms, "probes": args.probes,
                    "probes_meta": probes_art.get("meta"), "floors": floors,
                    "n_docs": args.n_docs, "feat_dim": 13,
                    "feat_layout": ["is_placeholder", "p6", "level_index", "n_levels",
                                    "log10_aset", "log10_floor", "type_oh(7)"]},
           "n_spans": len(spans),
           "n_flat_spans": sum(1 for s in spans if s["flat"]),
           "spans": spans, "assembly_bundles": bundles}
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print(f"{len(spans)} spans ({out['n_flat_spans']} flat), "
          f"{len(bundles)} doc bundles -> {OUT}")


if __name__ == "__main__":
    main()
