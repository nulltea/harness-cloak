"""Teacher-config A/B harness: prompt version x thinking mode, validated head-to-head.

Same clinical candidates, same cached anchors, same reader, same thresholds as the probe
builds — only the teacher configuration varies. Baseline (non-thinking, prompt v2) is read
from data/probes_validated.json (probe build 4). Nothing here writes the probe cache or the
validated artifact. Supersedes teacher_thinking_ab.py (thinking arm = --prompt 2 --thinking).

Run: INFERDPT_LLM_CACHE=data/llm_cache PYTHONPATH=src:scripts \
       .venv/bin/python -u scripts/spikes/teacher_config_ab.py --prompt 3 [--thinking]
       [--n-docs 16] [--workers 6]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from build_arms_artifact import load_artifact
from build_probes import validate_probes
from train_ranker import assemble

from cloak.corpora import load_task_docs, refs_of
from cloak.train.probes import PROMPT, TEACHER_MODEL, _parse_questions
from cloak.train.reward import canon, fact_f1s, restated_probes
from cloak.train.roundtrip import roundtrip_batch
from inferdpt.llm import LLMClient
from inferdpt.pipeline import pmap

# v3 (2026-07-05): targets the measured ceiling-failure taxonomy — full-gold context
# (cross-document ambiguity), whole-document uniqueness, extractive-grader awareness,
# span-type hint. v2 (cloak.train.probes.PROMPT) sees only the restating sentence.
PROMPT_V3 = """You are writing quiz questions used to check whether a specific fact survives in \
summaries of a document. The questions will be answered by a literal-minded extractive QA \
system that returns a short exact text span and abstains when a question is vague.

Document:
{gold}

Target fact: "{answer}" ({type_hint})

Write exactly THREE different short factual questions, one per line, such that:
- the ONLY correct answer anywhere in the document is "{answer}" — if other facts in the \
document could also answer the question, make the question more specific until they cannot;
- each question is answerable from the document alone;
- no question contains "{answer}" itself;
- each question expects a short exact answer, not a list or an explanation.

Reply with the three questions only, one per line."""

TYPE_HINT = {"PERSON": "a person's name", "LOC": "a location", "ORG": "an organization",
             "DATETIME": "a date or time", "QUANTITY": "a quantity or dose",
             "DEM": "a personal attribute"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", type=int, choices=(2, 3), default=3)
    ap.add_argument("--thinking", action="store_true",
                    help="omit enable_thinking:false (Qwen thinks; content = questions)")
    ap.add_argument("--n-docs", type=int, default=16)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--th", type=float, default=0.5)
    args = ap.parse_args()
    tag = f"p{args.prompt}-{'think' if args.thinking else 'nothink'}"

    art = load_artifact()
    env = json.loads(Path("data/ranker_env.json").read_text())
    docs = [d for d in load_task_docs("clinical", args.n_docs) if d["id"] in art["clinical"]]

    kw = dict(temperature=0.0, max_tokens=2048 if args.thinking else 512)
    if not args.thinking:
        kw["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    teacher = LLMClient(TEACHER_MODEL, base_url="http://localhost:8060/v1", api_key="x", **kw)

    cands = []
    for d in docs:
        gold = refs_of(d)[0]
        R = art["clinical"][d["id"]]["tau_walk"][1]
        for p in restated_probes(R, gold):
            e = p["entry"]
            prompt = (PROMPT.format(answer=e["surface"], sent=p["gold_sent"])
                      if args.prompt == 2 else
                      PROMPT_V3.format(gold=gold, answer=e["surface"],
                                       type_hint=TYPE_HINT.get(e.get("type"),
                                                               "a specific detail")))
            cands.append({"doc_id": d["id"], "surface": e["surface"], "prompt": prompt})
    replies = pmap(lambda t: teacher.generate(t["prompt"]), cands, workers=args.workers)

    per_doc, n_lost = defaultdict(list), 0
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

    jobs, meta = [], []
    for d in docs:
        if not per_doc[d["id"]]:
            continue
        spans = env["corpora"]["clinical"][d["id"]]["spans"]
        if not spans:
            continue
        ph_choice = {s["surface"].lower():
                     s["actions"][next(i for i, a in enumerate(s["actions"])
                                       if a["mode"] == "placeholder")] for s in spans}
        lo_doc, lo_R = assemble(d["text"], art["clinical"][d["id"]]["tau_walk"][1],
                                spans, ph_choice)
        jobs += [{"corpus": "clinical", "doc_p": d["text"], "R": [], "probes": []},
                 {"corpus": "clinical", "doc_p": lo_doc, "R": lo_R, "probes": []}]
        meta += [(d["id"], "hi"), (d["id"], "lo")]
    outs = roundtrip_batch(jobs, workers=args.workers)
    anchor = defaultdict(dict)
    for (did, kind), o in zip(meta, outs):
        anchor[did][kind] = o["out_final"]

    arm, per_q = {}, {"cand": 0, "rej_c": 0, "rej_f": 0}
    for d in docs:
        ps = per_doc[d["id"]]
        if not ps or "hi" not in anchor[d["id"]]:
            continue
        hi = fact_f1s(anchor[d["id"]]["hi"], ps)
        lo = fact_f1s(anchor[d["id"]]["lo"], ps)
        kept, rc, rf = validate_probes(ps, hi, lo, args.th)
        per_q["cand"] += len(ps)
        per_q["rej_c"] += len(rc)
        per_q["rej_f"] += len(rf)
        arm[d["id"]] = {"kept_q": len(kept),
                        "kept_facts": len({canon(p["surface"]) for p in kept})}

    base = {}
    pv = json.loads(Path("data/probes_validated.json").read_text())["docs"]
    for did, e in pv.items():
        if did.startswith(("aci/", "mts/")):
            ks = e["train"] + e["heldout"]
            base[did] = len({canon(p["surface"]) for p in ks})

    ids = sorted(set(arm) | set(base))
    rows = [{"doc": i, f"{tag}_facts": arm.get(i, {}).get("kept_facts", 0),
             "baseline_p2_nothink_facts": base.get(i, 0)} for i in ids]
    mean_arm = sum(r[f"{tag}_facts"] for r in rows) / max(len(rows), 1)
    mean_base = sum(r["baseline_p2_nothink_facts"] for r in rows) / max(len(rows), 1)
    report = {"arm": tag, "teacher": TEACHER_MODEL, "n_teacher_calls": len(cands),
              "lost_replies": n_lost,
              "ceiling_reject_rate": round(per_q["rej_c"] / max(per_q["cand"], 1), 3),
              "floor_reject_rate": round(per_q["rej_f"] / max(per_q["cand"], 1), 3),
              "kept_facts_mean": round(mean_arm, 2),
              "baseline_kept_facts_mean": round(mean_base, 2),
              "docs_ge4_facts": sum(r[f"{tag}_facts"] >= 4 for r in rows),
              "baseline_docs_ge4_facts": sum(r["baseline_p2_nothink_facts"] >= 4
                                             for r in rows),
              "rows": rows}
    out = Path(f"results/teacher_ab_{tag}.json")
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
