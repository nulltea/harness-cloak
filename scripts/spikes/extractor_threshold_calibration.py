"""Mini-calibration of frozen-extractor thresholds (diagnostic-grade forerunner of the
benchmark's calibration split — same discipline, smaller n: judge-proposed gold, coordinator
audit, fit on the calibration split ONCE, evaluate held-out ONCE; never tune on the eval).

Stages:
  --stage build  -> gold set data/extractor_calibration_set.jsonl (one row per residue entry
                    with gold mention span in the cascade prepass, or ABSENT) + audit dump
                    results/extractor_calibration_audit.txt (hand-check the grounded golds).
  --stage fit    -> two-pass sweep on the calibration split (ASSIGN_MARGIN at pinned NLI,
                    then NLI_ENTAIL at the chosen margin). Hard constraint: false splices = 0
                    (splice on an ABSENT-gold entry, or classify_recovery != 'recovered' at a
                    grounded gold). Maximize recovered; ties -> more conservative threshold.
                    NLI results are memoized per (premise, hypothesis) so the grid is cheap.
  --stage eval   -> run held-out split once at --margin/--nli, report per-type outcomes.

Run from the MAIN repo root (data/ lives there); code from the worktree:
  WT=.claude/worktrees/frozen-extractor
  CLOAK_LLM_CACHE=data/llm_cache PYTHONPATH=$WT/src:$WT/scripts:$WT/scripts/spikes \
    .venv/bin/python -u $WT/scripts/spikes/extractor_threshold_calibration.py --stage build \
    --env data/ranker_env_reconstructor_fine.json --arms data/task_arms_reconstructor_fine.json \
    --corpora clinical,lexsum --n-docs 80
"""
import argparse
import copy
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from survival_by_type import (build_jobs, _judge, parse_judge, grounded, SYSTEM, JUDGE_TMPL)
from cloak.extract import _rule_prepass
from cloak.reconstruct import classify_recovery
import cloak.frozen_extractor as fx
from cloak.train.roundtrip import roundtrip_batch

SET_PATH = Path("data/extractor_calibration_set.jsonl")
AUDIT_PATH = Path("results/extractor_calibration_audit.txt")
REPORT_PATH = Path("results/extractor_threshold_calibration.json")
SPLIT_SEED = "extractor-cal-v1"
CAL_FRAC = 0.5

MARGIN_GRID = [0.05, 0.03, 0.02, 0.01, 0.005, 0.0]
NLI_GRID = [0.80, 0.70, 0.60, 0.50, 0.40]


def _split(doc_id: str) -> str:
    h = int(hashlib.md5(f"{SPLIT_SEED}:{doc_id}".encode()).hexdigest(), 16) % 10_000
    return "cal" if h < CAL_FRAC * 10_000 else "heldout"


def _locate(quote: str, text: str):
    pat = re.compile(r"\s+".join(re.escape(w) for w in quote.split()), re.IGNORECASE)
    m = pat.search(text)
    return [m.start(), m.end()] if m else None


def stage_build(args):
    jobs, metas = build_jobs(args)
    outs = roundtrip_batch(jobs, workers=args.workers)
    judge = _judge()
    rows, audit = [], []
    for m, o in zip(metas, outs):
        out_p = o["out_p"]
        prepass, _, residue = _rule_prepass(out_p, m["R"], semantic=True)
        if not residue:
            continue
        items = "\n".join(
            f'{i}. "{e["surface"]}" -> "{e["replacement"]}"  [{e.get("type", "MISC")}]'
            for i, e in enumerate(residue))
        verdicts = parse_judge(judge.generate(JUDGE_TMPL.format(items=items, out_p=out_p),
                                              system=SYSTEM), len(residue))
        doc_row = {"corpus": m["corpus"], "doc_id": m["doc_id"], "split": _split(m["doc_id"]),
                   "doc_p": m["doc_p"], "R": m["R"], "out_p": out_p, "gold": []}
        for e, v in zip(residue, verdicts):
            q = v.get("quote")
            span = None
            if v.get("label") in ("SURVIVED", "REWORDED") and grounded(q, prepass):
                span = _locate(q, prepass)
            g = {"surface": e["surface"], "fill": e["replacement"],
                 "type": e.get("type", "MISC"),
                 "gold": {"span": span, "quote": q} if span else None}
            doc_row["gold"].append(g)
            if span:
                lo, hi = span
                audit.append(f"[{len(audit):02d}] {m['corpus']}/{m['doc_id']} ({doc_row['split']}) "
                             f"type={g['type']} fill={g['fill']!r} surface={g['surface']!r}\n"
                             f"     quote={q!r}\n"
                             f"     ctx: ...{prepass[max(0, lo-50):hi+50]}...\n")
        rows.append(doc_row)
    SET_PATH.write_text("\n".join(json.dumps(r) for r in rows))
    AUDIT_PATH.write_text(f"Grounded golds to hand-audit ({len(audit)}), edit the set file to "
                          f"null any bad gold before --stage fit\n\n" + "\n".join(audit))
    n_g = sum(1 for r in rows for g in r["gold"] if g["gold"])
    n_e = sum(len(r["gold"]) for r in rows)
    by_split = Counter(f"{r['split']}:{'g' if g['gold'] else 'a'}" for r in rows for g in r["gold"])
    print(f"{len(rows)} docs | {n_e} residue entries | {n_g} grounded golds | {dict(by_split)}")
    print(f"-> {SET_PATH}\n-> {AUDIT_PATH}")


class MemoNLI:
    """Wraps the real NLI callable with a (premise, hypothesis) memo so threshold sweeps
    re-decide from cached scores instead of re-running the model."""
    def __init__(self, nli):
        self.nli, self.memo = nli, {}
    def __call__(self, premise, hypothesis):
        k = (premise, hypothesis)
        if k not in self.memo:
            self.memo[k] = self.nli(premise, hypothesis)
        return self.memo[k]


def _run_split(rows, models, margin, nli_thr):
    pins = copy.deepcopy(fx.EXTRACTOR_PINS)
    pins["thresholds"]["ASSIGN_MARGIN"] = margin
    pins["thresholds"]["NLI_ENTAIL"] = nli_thr
    orig = fx.EXTRACTOR_PINS
    fx.EXTRACTOR_PINS = pins
    try:
        stats = Counter()
        per_type = {}
        for r in rows:
            prepass, _, _ = _rule_prepass(r["out_p"], r["R"], semantic=True)
            out_final, fstats = fx.extract(r["doc_p"], r["R"], r["out_p"], models=models)
            gold_by_surface = {g["surface"]: g for g in r["gold"]}
            for e in fstats.get("entries", []):
                g = gold_by_surface.get(e["surface"])
                if g is None:
                    continue
                t = per_type.setdefault(g["type"], Counter())
                if e["outcome"] == "spliced":
                    if g["gold"] is None:
                        stats["false_splice_absent"] += 1; t["false"] += 1
                    else:
                        cr = classify_recovery(out_final, g["gold"]["quote"],
                                               g["surface"], prepass)
                        if cr == "recovered":
                            stats["recovered"] += 1; t["recovered"] += 1
                        else:
                            stats[f"false_splice_{cr}"] += 1; t["false"] += 1
                else:
                    key = "abstain_had_gold" if g["gold"] else "abstain_correct"
                    stats[key] += 1
                    t[key] += 1
        stats["false_total"] = sum(v for k, v in stats.items() if k.startswith("false_splice"))
        return dict(stats), {k: dict(v) for k, v in per_type.items()}
    finally:
        fx.EXTRACTOR_PINS = orig


def stage_fit(args):
    rows = [json.loads(l) for l in SET_PATH.read_text().splitlines() if l.strip()]
    cal = [r for r in rows if r["split"] == "cal"]
    models = dict(fx.load_models(device=args.device))
    models["nli"] = MemoNLI(models["nli"])
    report = {"stage": "fit", "n_cal_docs": len(cal), "sweep": []}
    # pass 1: margin at pinned NLI
    base_nli = fx.EXTRACTOR_PINS["thresholds"]["NLI_ENTAIL"]
    best_margin, best = None, (-1, None)
    for mgn in MARGIN_GRID:
        s, _ = _run_split(cal, models, mgn, base_nli)
        report["sweep"].append({"ASSIGN_MARGIN": mgn, "NLI_ENTAIL": base_nli, **s})
        print(f"margin={mgn:<6} nli={base_nli}: {s}")
        if s.get("false_total", 0) == 0 and s.get("recovered", 0) > best[0]:
            best, best_margin = (s.get("recovered", 0), s), mgn
    if best_margin is None:
        best_margin = MARGIN_GRID[0]
    # pass 2: NLI at chosen margin
    best_nli = None; best = (-1, None)
    for thr in NLI_GRID:
        s, pt = _run_split(cal, models, best_margin, thr)
        report["sweep"].append({"ASSIGN_MARGIN": best_margin, "NLI_ENTAIL": thr, **s})
        print(f"margin={best_margin:<6} nli={thr}: {s}")
        if s.get("false_total", 0) == 0 and s.get("recovered", 0) > best[0]:
            best, best_nli = (s.get("recovered", 0), s), thr
    report["chosen"] = {"ASSIGN_MARGIN": best_margin, "NLI_ENTAIL": best_nli,
                        "note": "highest recovered at false_total=0; ties -> conservative"}
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["chosen"], indent=2))
    print(f"-> {REPORT_PATH}")


def stage_eval(args):
    rows = [json.loads(l) for l in SET_PATH.read_text().splitlines() if l.strip()]
    held = [r for r in rows if r["split"] == "heldout"]
    models = dict(fx.load_models(device=args.device))
    models["nli"] = MemoNLI(models["nli"])
    s, pt = _run_split(held, models, args.margin, args.nli)
    out = {"stage": "eval", "n_heldout_docs": len(held),
           "ASSIGN_MARGIN": args.margin, "NLI_ENTAIL": args.nli,
           "totals": s, "per_type": pt}
    prev = json.loads(REPORT_PATH.read_text()) if REPORT_PATH.exists() else {}
    prev["heldout_eval"] = out
    REPORT_PATH.write_text(json.dumps(prev, indent=2))
    print(json.dumps(out, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["build", "fit", "eval"], required=True)
    ap.add_argument("--env"); ap.add_argument("--arms")
    ap.add_argument("--corpora", default="clinical,lexsum")
    ap.add_argument("--n-docs", type=int, default=80)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--margin", type=float); ap.add_argument("--nli", type=float)
    args = ap.parse_args()
    if args.stage == "build":
        stage_build(args)
    elif args.stage == "fit":
        stage_fit(args)
    else:
        stage_eval(args)


if __name__ == "__main__":
    main()
