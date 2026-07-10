"""DIAGNOSTIC smoke — frozen extractor vs legacy cascade on real cached roundtrips.

NOT the benchmark: no oracle, no optimizer-fidelity gates, tiny n (see
docs/specs/extractor-frozen-rl-reward.md for the claim-bearing evaluation). Directional
only: on the fine-arms (doc_p, R, out_p) triples, run invert() and
frozen_extractor.extract() side by side and report (a) per-type residue outcomes under the
frozen ladder, (b) a do-no-harm cross-check — every original-surface occurrence the cascade
restored must also stand in the frozen output.

Run from the MAIN repo root (out_p comes from the warm content-addressed cache; no remote
calls when the arms/env match the last probe run):
  WT=.claude/worktrees/frozen-extractor
  CLOAK_LLM_CACHE=data/llm_cache PYTHONPATH=$WT/src:$WT/scripts:$WT/scripts/spikes \
    .venv/bin/python -u $WT/scripts/spikes/extractor_vs_cascade_smoke.py \
    --env data/ranker_env_reconstructor_fine.json \
    --arms data/task_arms_reconstructor_fine.json \
    --corpora clinical,lexsum --n-docs 40 --device cuda
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

from survival_by_type import build_jobs
from cloak.extract import invert
from cloak.frozen_extractor import extract, extractor_version, load_models
from cloak.train.roundtrip import roundtrip_batch

OUT = Path("results/extractor_vs_cascade_smoke.json")


def _count(surface: str, text: str) -> int:
    return len(re.findall(rf"(?<!\w){re.escape(surface)}(?!\w)", text, re.IGNORECASE))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True)
    ap.add_argument("--arms", required=True)
    ap.add_argument("--corpora", default="clinical,lexsum")
    ap.add_argument("--n-docs", type=int, default=40)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    jobs, metas = build_jobs(args)
    outs = roundtrip_batch(jobs, workers=args.workers)
    models = load_models(device=args.device)

    per_type = {}   # type -> Counter(outcome)
    tier0 = {"cascade_gen_absent": 0, "frozen_resolved_tier0": 0}
    harms, splices = [], []
    docs = 0

    for m, o in zip(metas, outs):
        out_p = o["out_p"]
        casc, cstats = invert(out_p, m["R"])
        frozen, fstats = extract(m["doc_p"], m["R"], out_p, models=models)
        docs += 1
        tier0["cascade_gen_absent"] += cstats.get("gen_absent", 0)
        tier0["frozen_resolved_tier0"] += fstats.get("resolved_tier0", 0)
        for e in fstats.get("entries", []):
            per_type.setdefault(e["type"], Counter())[f'{e["outcome"]}/{e.get("reason") or "-"}'] += 1
            if e["outcome"] == "spliced" and len(splices) < 25:
                i = frozen.lower().find(e["surface"].lower())
                splices.append({"corpus": m["corpus"], "doc": m["doc_id"],
                                "type": e["type"], "surface": e["surface"],
                                "ctx": frozen[max(0, i - 45):i + len(e["surface"]) + 45]})
        # do-no-harm: cascade restorations must survive in the frozen output
        for entry in m["R"]:
            s = entry["surface"]
            c_casc, c_froz = _count(s, casc), _count(s, frozen)
            if c_froz < c_casc:
                harms.append({"corpus": m["corpus"], "doc": m["doc_id"], "surface": s,
                              "cascade_n": c_casc, "frozen_n": c_froz})

    report = {
        "DIAGNOSTIC": "directional smoke, not the benchmark (no oracle / fidelity gates)",
        "settings": {**vars(args), "extractor_version": extractor_version()},
        "docs": docs,
        "tier0": tier0,
        "frozen_outcomes_by_type": {t: dict(c) for t, c in sorted(per_type.items())},
        "frozen_outcome_totals": dict(sum(per_type.values(), Counter())),
        "do_no_harm_violations": harms,   # MUST be []
        "example_splices": splices,
    }
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "example_splices"}, indent=2))
    print(f"-> {OUT}")
    if harms:
        raise SystemExit(f"DO-NO-HARM VIOLATIONS: {len(harms)}")


if __name__ == "__main__":
    main()
