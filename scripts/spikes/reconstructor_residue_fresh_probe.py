"""Fresh-data probe for the reconstructor design decision (no training, current pipeline).

Confirms/refutes with TODAY's fine-type detector+substitutor arms what the Jul-7 coarse
artifacts could not: (1) residue landscape, (2) judge quote-boundary overshoot, (3) admitted-
edit yield under three admission-gate variants, (4) zero-training ceiling — splice-at-quote
vs boundary-tight splice, scored per mention.

Stages run in one pass; per-doc work is: roundtrip (cached) -> cascade -> 1 judge call ->
local NLI/alignment. Outputs results/recon_fresh_probe.json + a hand-audit dump.

Run: CLOAK_LLM_CACHE=data/llm_cache PYTHONPATH=src:scripts:scripts/spikes \
       .venv/bin/python -u scripts/spikes/reconstructor_residue_fresh_probe.py \
       --env data/ranker_env_reconstructor_fine.json \
       --arms data/task_arms_reconstructor_fine.json \
       --corpora clinical,lexsum --n-docs 40 --workers 6
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

from rapidfuzz import fuzz

from survival_by_type import (build_jobs, _judge, parse_judge, grounded, SYSTEM, JUDGE_TMPL)
from cloak.extract import _rule_prepass, _type_sane
from cloak.reconstruct import (restorable, _corresponds, _load_nli, _norm,
                               splice_at_quote, classify_recovery)
from cloak.runtime_types import PLACEHOLDER_RE
from cloak.train.roundtrip import roundtrip_batch

OUT = Path("results/recon_fresh_probe.json")
AUDIT = Path("results/recon_fresh_admit_audit.txt")


def _best_align(quote: str, fill: str, surface: str):
    """Best (score, lo, hi) alignment of fill or surface inside the quote."""
    best = None
    for probe in (fill, surface):
        if not probe:
            continue
        al = fuzz.partial_ratio_alignment(probe.lower(), quote.lower())
        if al and al.score >= 60 and (best is None or al.score > best[0]):
            best = (al.score, al.dest_start, al.dest_end)
    return best


def overshoot_words(quote: str, fill: str, surface: str):
    """Words in the quote NOT covered by the best fill/surface alignment (superset noise).
    None = no alignment at all (quote is a different phrase entirely)."""
    al = _best_align(quote, fill, surface)
    if al is None:
        return None
    covered = len(quote[al[1]:al[2]].split())
    return max(0, len(quote.split()) - covered)


def tight_splice(prepass: str, quote: str, fill: str, surface: str):
    """Boundary-tight splice: locate quote in prepass, align fill/surface INSIDE it, replace
    only the aligned sub-span (snapped to word boundaries). Abstain when alignment fails."""
    pat = re.compile(r"\s+".join(re.escape(w) for w in quote.split()), re.IGNORECASE)
    m = pat.search(prepass)
    if not m:
        return prepass, "quote-not-found"
    qtext = prepass[m.start():m.end()]
    al = _best_align(qtext, fill, surface)
    if al is None:
        return prepass, "no-align-abstain"
    lo, hi = al[1], al[2]
    while lo > 0 and qtext[lo - 1].isalnum():
        lo -= 1
    while hi < len(qtext) and qtext[hi].isalnum():
        hi += 1
    return prepass[:m.start() + lo] + surface + prepass[m.start() + hi:], "spliced"


def gate_variant(e: dict, v: dict, prepass: str, nli, mode: str) -> bool:
    """current = restorable as shipped; corr_all = correspondence mandatory for EVERY type
    (keeps _type_sane); fixed = nonempty fill + placeholder guard + correspondence for all,
    WITHOUT the word-list _type_sane legs (probe showed they reject real fine-type mentions
    and always-pass MISC)."""
    q = v.get("quote")
    if v.get("label") not in ("SURVIVED", "REWORDED") or not q:
        return False
    if _norm(q) not in _norm(prepass):
        return False
    if mode == "current":
        return restorable(e, v, prepass, nli=nli)
    fill = (e.get("replacement") or "").strip()
    if not fill or PLACEHOLDER_RE.fullmatch(fill):
        return False
    if mode == "corr_all" and not _type_sane(e.get("type", "MISC"), fill, q):
        return False
    return _corresponds(fill, q, nli)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True)
    ap.add_argument("--arms", required=True)
    ap.add_argument("--corpora", default="clinical,lexsum")
    ap.add_argument("--n-docs", type=int, default=40)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    jobs, metas = build_jobs(args)
    outs = roundtrip_batch(jobs, workers=args.workers)
    judge = _judge()
    nli = _load_nli()

    stage1 = {}   # corpus -> residue landscape
    stage2 = {"overshoot_hist": Counter(), "no_align": 0, "grounded_quotes": 0, "examples": []}
    stage3 = {m: Counter() for m in ("current", "corr_all", "fixed")}
    stage4 = {"quote_splice": Counter(), "tight_splice": Counter(),
              "damage_words_quote": 0, "damage_words_tight": 0, "examples": []}
    audit_lines = []

    for m, o in zip(metas, outs):
        out_p = o["out_p"]
        prepass, _, residue = _rule_prepass(out_p, m["R"], semantic=True)
        s1 = stage1.setdefault(m["corpus"], {
            "docs": 0, "docs_with_residue": 0, "generalizations": 0,
            "residue": 0, "residue_by_type": Counter()})
        s1["docs"] += 1
        s1["generalizations"] += sum(1 for e in m["R"] if e["action"] == "generalize")
        if not residue:
            continue
        s1["docs_with_residue"] += 1
        s1["residue"] += len(residue)
        for e in residue:
            s1["residue_by_type"][e.get("type", "MISC")] += 1

        items = "\n".join(
            f'{i}. "{e["surface"]}" -> "{e["replacement"]}"  [{e.get("type", "MISC")}]'
            for i, e in enumerate(residue))
        verdicts = parse_judge(judge.generate(JUDGE_TMPL.format(items=items, out_p=out_p),
                                              system=SYSTEM), len(residue))

        for e, v in zip(residue, verdicts):
            q = v.get("quote")
            if not (v.get("label") in ("SURVIVED", "REWORDED") and grounded(q, prepass)):
                continue
            # stage 2: quote-boundary overshoot
            stage2["grounded_quotes"] += 1
            ov = overshoot_words(q, e["replacement"], e["surface"])
            if ov is None:
                stage2["no_align"] += 1
            else:
                stage2["overshoot_hist"][min(ov, 5)] += 1
                if ov >= 2 and len(stage2["examples"]) < 15:
                    stage2["examples"].append({"type": e.get("type", "MISC"),
                                               "fill": e["replacement"],
                                               "surface": e["surface"], "quote": q})
            # stage 3: gate variants
            admits = {}
            for mode in stage3:
                admits[mode] = gate_variant(e, v, prepass, nli, mode)
                if admits[mode]:
                    stage3[mode][e.get("type", "MISC")] += 1
            # stage 4: ceiling probe on the fixed-gate admitted set
            if admits["fixed"]:
                qs, ok = splice_at_quote(prepass, q, e["surface"])
                ts, how = tight_splice(prepass, q, e["replacement"], e["surface"])
                rq = classify_recovery(qs, q, e["surface"], prepass) if ok else "abstain"
                rt = classify_recovery(ts, q, e["surface"], prepass) \
                    if how == "spliced" else "abstain"
                stage4["quote_splice"][rq] += 1
                stage4["tight_splice"][rt] += 1
                ov2 = ov if ov is not None else len(q.split())
                stage4["damage_words_quote"] += ov2      # words a quote-splice would delete
                if len(stage4["examples"]) < 40:
                    i = _norm(prepass).find(_norm(q))
                    ctx = _norm(prepass)[max(0, i - 50):i + len(_norm(q)) + 50]
                    stage4["examples"].append({
                        "doc": m["doc_id"], "type": e.get("type", "MISC"),
                        "fill": e["replacement"], "surface": e["surface"], "quote": q,
                        "quote_splice": rq, "tight_splice": rt, "overshoot_words": ov})
                    audit_lines.append(
                        f"[{len(stage4['examples']) - 1:02d}] {m['corpus']}/{m['doc_id']} "
                        f"type={e.get('type', 'MISC')} fill={e['replacement']!r} "
                        f"surface={e['surface']!r}\n     quote={q!r} overshoot={ov}\n"
                        f"     ctx: ...{ctx}...\n"
                        f"     quote_splice={rq} tight_splice={rt} ({how})\n")

    for s1 in stage1.values():
        s1["residue_by_type"] = dict(s1["residue_by_type"])
    report = {
        "settings": vars(args),
        "stage1_residue_landscape": stage1,
        "stage2_quote_overshoot": {
            "grounded_quotes": stage2["grounded_quotes"],
            "no_align": stage2["no_align"],
            "overshoot_words_hist(cap5)": dict(sorted(stage2["overshoot_hist"].items())),
            "examples_overshoot_ge2": stage2["examples"]},
        "stage3_gate_admits_by_type": {k: dict(v) for k, v in stage3.items()},
        "stage3_gate_totals": {k: sum(v.values()) for k, v in stage3.items()},
        "stage4_ceiling": {
            "quote_splice_outcomes": dict(stage4["quote_splice"]),
            "tight_splice_outcomes": dict(stage4["tight_splice"]),
            "nontarget_words_quote_splice_would_delete": stage4["damage_words_quote"],
            "examples": stage4["examples"]},
    }
    OUT.write_text(json.dumps(report, indent=2))
    AUDIT.write_text(f"Fresh admitted-edit audit ({len(audit_lines)} cases)\n\n"
                     + "\n".join(audit_lines))
    print(json.dumps({k: v for k, v in report.items() if k != "stage4_ceiling"}, indent=2))
    print(json.dumps({k: v for k, v in report["stage4_ceiling"].items()
                      if k != "examples"}, indent=2))
    print(f"-> {OUT}\n-> {AUDIT}")


if __name__ == "__main__":
    main()
