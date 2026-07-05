"""A/B: does Qwen3.6 WITH thinking write better probe questions than non-thinking?

Same candidates (clinical restated surfaces), same 3-question prompt, same cached anchors,
same reader, same validation thresholds — only the teacher's thinking mode differs. The
non-thinking arm is read from the current data/probes_validated.json (probe build 4); the
thinking arm is generated here (LLM disk cache keys include params, so no collisions) and
validated out-of-band (no writes to the probe cache or validated artifact).

Run: INFERDPT_LLM_CACHE=data/llm_cache PYTHONPATH=src:scripts \
       .venv/bin/python -u scripts/spikes/teacher_thinking_ab.py [--n-docs 16] [--workers 6]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from build_arms_artifact import load_artifact
from build_probes import validate_probes

from cloak.corpora import load_task_docs, refs_of
from cloak.train.probes import PROMPT, TEACHER_MODEL, _parse_questions
from cloak.train.reward import canon, fact_f1s, restated_probes
from cloak.train.roundtrip import roundtrip_batch
from inferdpt.llm import LLMClient
from inferdpt.pipeline import pmap

OUT = Path("results/teacher_thinking_ab.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-docs", type=int, default=16)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--th", type=float, default=0.5)
    args = ap.parse_args()

    art = load_artifact()
    docs = [d for d in load_task_docs("clinical", args.n_docs) if d["id"] in art["clinical"]]
    # thinking ON: no enable_thinking kwarg (default); budget covers reasoning + questions
    teacher = LLMClient(TEACHER_MODEL, base_url="http://localhost:8060/v1", api_key="x",
                        temperature=0.0, max_tokens=2048)

    cands = []
    for d in docs:
        R = art["clinical"][d["id"]]["tau_walk"][1]
        for p in restated_probes(R, refs_of(d)[0]):
            cands.append({"doc_id": d["id"], "surface": p["entry"]["surface"],
                          "sent": p["gold_sent"]})
    replies = pmap(lambda t: teacher.generate(
        PROMPT.format(answer=t["surface"], sent=t["sent"])), cands, workers=args.workers)

    per_doc = defaultdict(list)
    n_lost = 0
    for t, r in zip(cands, replies):
        reply = (r or "").strip()
        if not reply or "<think>" in reply:
            n_lost += 1
            continue
        qs = _parse_questions(reply, t["surface"])
        if not qs:
            n_lost += 1
            continue
        for q in qs:
            per_doc[t["doc_id"]].append({"surface": t["surface"], "question": q})

    # anchors (cache hits from the probe builds) + per-question validation
    jobs, meta = [], []
    for d in docs:
        if not per_doc[d["id"]]:
            continue
        for kind, doc_p, R in (("hi", d["text"], []),
                               (None, None, None),):
            if kind is None:
                break
            jobs.append({"corpus": "clinical", "doc_p": doc_p, "R": R, "probes": []})
            meta.append((d["id"], "hi"))
    # floor anchors need the all-placeholder doc — reuse build_probes' construction
    env = json.loads(Path("data/ranker_env.json").read_text())
    from train_ranker import assemble
    for d in docs:
        if not per_doc[d["id"]]:
            continue
        spans = env["corpora"]["clinical"][d["id"]]["spans"]
        if not spans:
            continue
        ph_idx = {s["surface"].lower():
                  s["actions"][next(i for i, a in enumerate(s["actions"])
                                    if a["mode"] == "placeholder")] for s in spans}
        lo_doc, lo_R = assemble(d["text"], art["clinical"][d["id"]]["tau_walk"][1],
                                spans, ph_idx)
        jobs.append({"corpus": "clinical", "doc_p": lo_doc, "R": lo_R, "probes": []})
        meta.append((d["id"], "lo"))
    outs = roundtrip_batch(jobs, workers=args.workers)
    anchor = defaultdict(dict)
    for (did, kind), o in zip(meta, outs):
        anchor[did][kind] = o["out_final"]

    think = {}
    for d in docs:
        ps = per_doc[d["id"]]
        if not ps or "hi" not in anchor[d["id"]] or "lo" not in anchor[d["id"]]:
            continue
        hi = fact_f1s(anchor[d["id"]]["hi"], ps)
        lo = fact_f1s(anchor[d["id"]]["lo"], ps)
        kept, _, _ = validate_probes(ps, hi, lo, args.th)
        think[d["id"]] = {"kept_q": len(kept),
                          "kept_facts": len({canon(p["surface"]) for p in kept})}

    nonthink = {}
    pv = json.loads(Path("data/probes_validated.json").read_text())["docs"]
    for did, e in pv.items():
        if not did.startswith(("aci/", "mts/")):
            continue
        ks = e["train"] + e["heldout"]
        nonthink[did] = {"kept_q": len(ks),
                         "kept_facts": len({canon(p["surface"]) for p in ks})}

    ids = sorted(set(think) | set(nonthink))
    rows = [{"doc": i,
             "think_facts": think.get(i, {}).get("kept_facts", 0),
             "nonthink_facts": nonthink.get(i, {}).get("kept_facts", 0)} for i in ids]
    tf = sum(r["think_facts"] for r in rows) / max(len(rows), 1)
    nf = sum(r["nonthink_facts"] for r in rows) / max(len(rows), 1)
    report = {"teacher": TEACHER_MODEL, "n_candidates": len(cands), "lost_replies": n_lost,
              "thinking_kept_facts_mean": round(tf, 2),
              "nonthinking_kept_facts_mean": round(nf, 2), "rows": rows}
    OUT.write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=1))
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
