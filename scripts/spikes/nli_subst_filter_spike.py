"""Spike: NLI v3 binary SUBSTITUTION-FRAME type verification as a detector-noise filter.

Throwaway probe (scripts/spikes/). Successor to the v2 self-type spike
(nli_selftype_filter_spike.py).

WHY v3 (the only real change): v2 used the ill-posed ASSERTION frame -- premise = the span's
sentence, hypothesis = "<Surface> is <a type>." A usage sentence never asserts an entity's
CATEGORY, so entailment there is uninformative (measured AUC 0.677, unusable at every threshold).
v3 uses the WELL-POSED SUBSTITUTION frame instead: does the original sentence entail the sentence
with the span REPLACED by "a <type>"? That is exactly what cloak.lattice.nli_gate_batch already
does to certify generalizations -- so we REUSE it directly rather than a hand-rolled scorer.

For each detected span we form exactly ONE job = (surface, sentence, ["a <own-type> phrase"]) and
run nli_gate_batch(jobs, thresh=0.0) so we read the raw entailment of the self-type substitution for
every span (0.0 = keep the score; we threshold ourselves in the sweep). A span whose own-type phrase
is degenerate/self-referential is dropped by nli_gate_batch's _nli_prep -> empty result -> we record
ent=0.0 (fail-closed, drop-eligible) and COUNT it as n_prep_filtered. KEEP iff ent >= tau else DROP.
No retype in this binary spike (retype = follow-up).

Calibration is REPRESENTATIVE (same as v2), drawn from real detected spans / real sentences:
  - eval-KEEP = spans whose _norm surface hits its own type in the lattice profiles (lookup_entry).
  - eval-DROP = spans whose _norm surface is in a detector deny-list junk set.
tau is calibrated so eval-KEEP false-drop-rate stays low while eval-DROP recall is maximized.

GPU: the real run loads the NLI model + clinical corpus. Run --smoke (model-free, corpus-free,
nli_gate_batch faked) for the runnable check.

    PYTHONPATH=src:scripts .venv/bin/python scripts/spikes/nli_subst_filter_spike.py --smoke
    PYTHONPATH=src:scripts .venv/bin/python scripts/spikes/nli_subst_filter_spike.py         # real
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from cloak.lattice import NLI_MODEL

SPANS_PATH = Path("results/mined_lattice_profile_spans_large.jsonl")
OUT_PATH = Path("results/nli_subst_spike.json")
DETECTOR_MS_PER_DOC = 324.0  # measured detector reference

# Short GRAMMATICAL noun phrases that can substitute for the span in its sentence (v3 frame).
# A span whose type has no phrase here is skipped entirely.
TYPE_SUB_PHRASES = {
    "health-condition": "a medical condition",
    "medical-procedure": "a medical procedure",
    "drug": "a medication",
    "injury": "an injury",
    "organization-medical-facility": "a medical facility",
    "LOC": "a location",
    "profession": "an occupation",
    "nationality": "a nationality",
    "religion": "a religion",
    "ethnicity": "an ethnicity",
}


# ---------- scoring (gate-injected; shared by real + smoke) ----------

def score_subst(spans_meta: list[dict], gate) -> dict:
    """Attach `ent` (entailment of the span's self-type SUBSTITUTION) to each span in place, via
    ONE nli_gate_batch job per span at thresh=0.0. Spans whose orig_type has no phrase get ent=None
    (excluded downstream). Spans whose job returns empty (nli_gate_batch's _nli_prep filtered the
    self-type phrase: self-ref / no sentence / degenerate dup) get ent=0.0 and are counted as
    n_prep_filtered (fail-closed -> drop-eligible). `gate` maps
    jobs=[(entity, context, [candidate])] -> per-job list of (candidate, entailment_score)."""
    jobs: list[tuple[str, str, list[str]]] = []
    idx: list[int] = []
    for i, m in enumerate(spans_meta):
        phrase = TYPE_SUB_PHRASES.get(m["orig_type"])
        if phrase is None:
            m["ent"] = None
            continue
        jobs.append((m["surface"], m["premise"], [phrase]))
        idx.append(i)
    t0 = time.time()
    approved = gate(jobs, 0.0)  # thresh=0.0 -> every surviving pair's score is returned
    elapsed_ms = (time.time() - t0) * 1000.0
    assert len(approved) == len(jobs), f"gate returned {len(approved)} for {len(jobs)} jobs"
    n_prep_filtered = 0
    for j, i in enumerate(idx):
        res = approved[j]
        if res:
            spans_meta[i]["ent"] = float(res[0][1])
        else:  # _nli_prep filtered the self-type phrase -> fail closed to a drop-eligible score
            spans_meta[i]["ent"] = 0.0
            n_prep_filtered += 1
    return {"n_scored": len(jobs), "nli_ms": elapsed_ms, "n_prep_filtered": n_prep_filtered}


# ---------- calibration / metrics ----------

def calibrate(keep_ents: list[float], drop_ents: list[float], fdr_bar: float):
    """Largest tau with eval-KEEP false-drop-rate <= fdr_bar (maximizes eval-DROP recall, which is
    monotone in tau). Returns (tau, drop_recall, keep_false_drop_rate) or None if infeasible."""
    if not keep_ents:
        return None
    cands = sorted(set(keep_ents + drop_ents))
    best = None
    for tau in cands:
        fdr = sum(e < tau for e in keep_ents) / len(keep_ents)
        if fdr <= fdr_bar:
            rec = (sum(e < tau for e in drop_ents) / len(drop_ents)) if drop_ents else 0.0
            if best is None or rec > best[1] or (rec == best[1] and tau > best[0]):
                best = (tau, rec, fdr)
    return best


def youden(keep_ents: list[float], drop_ents: list[float]):
    """tau maximizing Youden J = drop_recall - keep_false_drop_rate (TPR - FPR)."""
    if not keep_ents or not drop_ents:
        return None
    best = None
    for tau in sorted(set(keep_ents + drop_ents)):
        tpr = sum(e < tau for e in drop_ents) / len(drop_ents)
        fpr = sum(e < tau for e in keep_ents) / len(keep_ents)
        j = tpr - fpr
        if best is None or j > best["youden_j"]:
            best = {"tau": round(tau, 4), "youden_j": round(j, 4),
                    "drop_recall": round(tpr, 4), "keep_false_drop_rate": round(fpr, 4)}
    return best


def auc_junk_vs_real(keep_ents: list[float], drop_ents: list[float]):
    """AUC of ent separating eval-DROP (junk) from eval-KEEP (real). The pairwise win-rate below
    equals the trapezoidal ROC AUC exactly (Mann-Whitney U / |keep|*|drop|), ties count 0.5.
    ponytail: O(|keep|*|drop|); a few hundred spans, no need for the sort-based version."""
    if not keep_ents or not drop_ents:
        return None
    wins = sum((k > d) + 0.5 * (k == d) for k in keep_ents for d in drop_ents)
    return wins / (len(keep_ents) * len(drop_ents))


def sweep(keep_ents: list[float], drop_ents: list[float]) -> list[dict]:
    rows = []
    cands = sorted(set(keep_ents + drop_ents))
    for tau in cands:
        kd = sum(e < tau for e in keep_ents)
        dd = sum(e < tau for e in drop_ents)
        fdr = kd / len(keep_ents) if keep_ents else None
        rec = dd / len(drop_ents) if drop_ents else None
        prec = dd / (dd + kd) if (dd + kd) else None
        rows.append({"tau": round(tau, 4),
                     "keep_false_drop_rate": None if fdr is None else round(fdr, 4),
                     "drop_recall": None if rec is None else round(rec, 4),
                     "rejection_precision": None if prec is None else round(prec, 4)})
    return rows


OP_BARS = [0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50]  # max real-loss (eval-KEEP false-drop) bars


def operating_points(keep_ents: list[float], drop_ents: list[float]) -> list[dict]:
    rows = []
    for bar in OP_BARS:
        c = calibrate(keep_ents, drop_ents, bar)
        rows.append({"max_real_loss": bar,
                     "tau": round(c[0], 4) if c else None,
                     "junk_dropped": round(c[1], 4) if c else None,
                     "real_loss": round(c[2], 4) if c else None})
    return rows


# ---------- real corpus wiring ----------

def load_real_spans() -> tuple[list[dict], int, int]:
    """Returns (spans_meta, n_no_context, n_docs). Each row: {surface (normed), orig_type, premise,
    eval in {'keep','drop',None}}. eval-KEEP = own-type lookup_entry hit; eval-DROP = _norm surface
    in a detector deny-list junk set; None = unlabeled (residue)."""
    from build_mined_lattice_profiles import (DetectedSpan, _unique_spans,
                                              normalize_detector_label)
    from build_mined_lattice_profiles import _norm as norm_surface
    from cloak.corpora import load_task_docs
    from cloak.detect import (_NOISE_ANATOMY, _NOISE_DEVICE_SUPPLIES,
                              _NOISE_IMAGING_DIAGNOSTICS, _NOISE_LAB_TESTS)
    from cloak.lattice_profiles import lookup_entry
    from cloak.train.ladder_probes import sentence_of

    denylist = set().union(_NOISE_LAB_TESTS, _NOISE_IMAGING_DIAGNOSTICS,
                           _NOISE_ANATOMY, _NOISE_DEVICE_SUPPLIES)

    raw = [json.loads(line) for line in SPANS_PATH.read_text().splitlines() if line.strip()]
    spans = [DetectedSpan(r["surface"], r["detector_label"], r["doc_id"], float(r["score"]))
             for r in raw]
    unique = _unique_spans(spans)  # surface already _norm'd inside

    docs = load_task_docs("clinical")
    text_of = {d["id"]: d["text"] for d in docs}

    meta: list[dict] = []
    n_no_context = 0
    for sp in unique:
        text = text_of.get(sp.doc_id)
        premise = sentence_of(text, sp.surface) if text else ""
        if not premise:
            n_no_context += 1
            continue
        rt = normalize_detector_label(sp.detector_label)
        surf = norm_surface(sp.surface)
        if lookup_entry(surf, rt) is not None:
            ev = "keep"
        elif surf in denylist:
            ev = "drop"
        else:
            ev = None
        meta.append({"surface": surf, "orig_type": rt, "premise": premise, "eval": ev})
    return meta, n_no_context, len(docs)


def margin_action_of(spans_meta: list[dict]):
    from cloak.profile_match import span_key
    from cloak.span_gate import gate_spans
    items = [(m["surface"], m["orig_type"]) for m in spans_meta]
    decisions = gate_spans(items, "miner")
    out: dict[tuple[str, str], str] = {}
    for m in spans_meta:
        k = span_key(m["surface"], m["orig_type"])
        d = decisions.get(k)
        if d is None:
            out[k] = "none"
        elif d.action == "drop":
            out[k] = "drop-" + d.layer
        else:
            out[k] = d.action  # keep / retype
    return out, span_key


# ---------- smoke ----------

# Synthetic spans carry an explicit eval label + a fake self-type-substitution entailment (_ent).
# The fake nli_gate_batch keys on the surface. No model, no corpus. 'PREP' marks a span the fake
# gate filters (empty result) to exercise n_prep_filtered -> ent=0.0 fail-closed.
SMOKE_SPANS = [
    # eval-KEEP: original sentence entails the self-type substitution
    {"surface": "blorbitis", "orig_type": "health-condition", "eval": "keep", "_ent": 0.95},
    {"surface": "ankleoid", "orig_type": "injury", "eval": "keep", "_ent": 0.88},
    {"surface": "zorbivex", "orig_type": "drug", "eval": "keep", "_ent": 0.90},
    {"surface": "clinoplace", "orig_type": "organization-medical-facility", "eval": "keep", "_ent": 0.85},
    {"surface": "scanwork", "orig_type": "medical-procedure", "eval": "keep", "_ent": 0.82},
    {"surface": "coastville", "orig_type": "LOC", "eval": "keep", "_ent": 0.80},
    # eval-DROP: junk; substitution not entailed
    {"surface": "brickthing", "orig_type": "drug", "eval": "drop", "_ent": 0.10},
    {"surface": "woodplank", "orig_type": "medical-procedure", "eval": "drop", "_ent": 0.15},
    {"surface": "greyish", "orig_type": "health-condition", "eval": "drop", "_ent": 0.20},
    {"surface": "blankword", "orig_type": "injury", "eval": "drop", "_ent": 0.05},
    {"surface": "noiseblob", "orig_type": "organization-medical-facility", "eval": "PREP"},  # prep-filtered
    # unlabeled residue: only used in the applied run
    {"surface": "midword", "orig_type": "drug", "eval": None, "_ent": 0.55},
    {"surface": "otherword", "orig_type": "health-condition", "eval": None, "_ent": 0.40},
    # no-phrase type: skipped entirely (ent=None)
    {"surface": "someword", "orig_type": "gender", "eval": None, "_ent": 0.99},
]


def smoke_gate_factory(spans_meta):
    """Fake nli_gate_batch: (jobs, thresh) -> per-job [(candidate, ent)]; [] for 'PREP' surfaces."""
    ent_of = {m["surface"]: m.get("_ent") for m in spans_meta}
    prep = {m["surface"] for m in spans_meta if m["eval"] == "PREP"}

    def gate(jobs, thresh):
        out = []
        for entity, _ctx, cands in jobs:
            if entity in prep:
                out.append([])
            else:
                e = ent_of.get(entity, 0.1)
                out.append([(cands[0], e)] if e >= thresh else [])
        return out

    return gate


def run_smoke() -> int:
    spans_meta = [{"surface": m["surface"], "orig_type": m["orig_type"],
                   "premise": f"The chart says {m['surface']} was noted.",
                   "eval": None if m["eval"] == "PREP" else m["eval"], "_ent": m.get("_ent")}
                  for m in SMOKE_SPANS]
    gate = smoke_gate_factory(SMOKE_SPANS)
    meta = score_subst(spans_meta, gate)

    scored = [m for m in spans_meta if m["ent"] is not None]
    assert len(scored) == len(SMOKE_SPANS) - 1, "the no-phrase span should be skipped"
    assert meta["n_prep_filtered"] == 1, "the PREP span should fail closed (ent=0.0, counted)"
    keep_ents = [m["ent"] for m in scored if m["eval"] == "keep"]
    drop_ents = [m["ent"] for m in scored if m["eval"] == "drop"]

    auc = auc_junk_vs_real(keep_ents, drop_ents)
    assert auc is not None and 0.0 <= auc <= 1.0, f"AUC did not compute: {auc}"

    chosen = calibrate(keep_ents, drop_ents, 0.05)
    assert chosen is not None, "calibration failed to pick a finite tau"
    tau = chosen[0]
    kept = [m for m in scored if m["ent"] >= tau]
    dropped = [m for m in scored if m["ent"] < tau]
    assert kept, "no spans kept at chosen tau"
    assert dropped, "no spans dropped at chosen tau"
    assert youden(keep_ents, drop_ents) is not None

    print(f"SMOKE scored={len(scored)} eval_keep={len(keep_ents)} eval_drop={len(drop_ents)} "
          f"prep_filtered={meta['n_prep_filtered']}")
    print(f"SMOKE auc={auc:.3f} chosen_tau={tau:.3f} drop_recall={chosen[1]:.3f} keep_fdr={chosen[2]:.3f}")
    print(f"SMOKE applied: keep={len(kept)} drop={len(dropped)}")
    print("SMOKE OK: keep AND drop both occur; AUC finite; calibration picked a finite tau")
    return 0


# ---------- main (real) ----------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        return run_smoke()

    from cloak.lattice import nli_gate_batch

    spans_meta, n_no_context, n_docs = load_real_spans()
    meta = score_subst(spans_meta, nli_gate_batch)
    scored = [m for m in spans_meta if m["ent"] is not None]
    n_scored, nli_ms, n_prep_filtered = meta["n_scored"], meta["nli_ms"], meta["n_prep_filtered"]

    keep_ents = [m["ent"] for m in scored if m["eval"] == "keep"]
    drop_ents = [m["ent"] for m in scored if m["eval"] == "drop"]
    n_residue = sum(1 for m in scored if m["eval"] is None)

    auc = auc_junk_vs_real(keep_ents, drop_ents)
    chosen = calibrate(keep_ents, drop_ents, 0.05)
    tau_002 = calibrate(keep_ents, drop_ents, 0.02)
    tau_010 = calibrate(keep_ents, drop_ents, 0.10)
    chosen_tau = chosen[0] if chosen else None
    best_youden = youden(keep_ents, drop_ents)
    ops = operating_points(keep_ents, drop_ents)

    # per-type tau only where the eval sets are large enough, else global
    per_type = {}
    types = {m["orig_type"] for m in scored}
    for t in sorted(types):
        ke = [m["ent"] for m in scored if m["orig_type"] == t and m["eval"] == "keep"]
        de = [m["ent"] for m in scored if m["orig_type"] == t and m["eval"] == "drop"]
        if len(ke) >= 10 and len(de) >= 5:
            c = calibrate(ke, de, 0.05)
            per_type[t] = {"tau": round(c[0], 4), "drop_recall": round(c[1], 4),
                           "n_keep": len(ke), "n_drop": len(de)} if c else "infeasible"
        else:
            per_type[t] = "global"

    # rejection precision + drop recall at CHOSEN tau
    if chosen_tau is not None:
        kd = sum(e < chosen_tau for e in keep_ents)
        dd = sum(e < chosen_tau for e in drop_ents)
        rejection_precision = dd / (dd + kd) if (dd + kd) else None
        drop_recall = dd / len(drop_ents) if drop_ents else None
        keep_false_drop_rate = kd / len(keep_ents) if keep_ents else None
    else:
        rejection_precision = drop_recall = keep_false_drop_rate = None

    # APPLY at chosen global tau + head-to-head vs the margin layer
    for m in scored:
        m["nli_action"] = "keep" if (chosen_tau is not None and m["ent"] >= chosen_tau) else "drop"

    margins, span_key = margin_action_of(scored)

    def m_act(m):
        return margins.get(span_key(m["surface"], m["orig_type"]), "none")

    crosstab = {"nli_keep": 0, "nli_drop": 0}
    margin_dropped_but_v3_kept = []
    v3_dropped_but_margin_kept = []
    for m in scored:
        ma = m_act(m)
        if ma == "drop-margin":
            crosstab["nli_" + m["nli_action"]] += 1
            if m["nli_action"] == "keep":
                margin_dropped_but_v3_kept.append(
                    {"surface": m["surface"], "type": m["orig_type"], "ent": round(m["ent"], 4)})
        if m["nli_action"] == "drop" and ma not in ("drop-margin", "drop-denylist", "none"):
            v3_dropped_but_margin_kept.append(
                {"surface": m["surface"], "type": m["orig_type"], "ent": round(m["ent"], 4)})

    v3_dropped = [{"surface": m["surface"], "type": m["orig_type"], "ent": round(m["ent"], 4)}
                  for m in scored if m["nli_action"] == "drop"][:40]

    def act_counts(key):
        out: dict[str, int] = {}
        for m in scored:
            out[m[key]] = out.get(m[key], 0) + 1
        return dict(sorted(out.items()))

    margin_counts: dict[str, int] = {}
    for m in scored:
        a = m_act(m)
        margin_counts[a] = margin_counts.get(a, 0) + 1

    # COST -- nli_gate_batch is 1 pair/span (same as v2's self-type scorer)
    spans_per_doc = n_scored / n_docs if n_docs else 0.0
    ms_per_span = nli_ms / n_scored if n_scored else 0.0
    ms_per_doc_all = ms_per_span * spans_per_doc
    residue_frac = n_residue / n_scored if n_scored else 0.0
    ms_per_doc_residue = ms_per_doc_all * residue_frac
    ratio_all = ms_per_doc_all / DETECTOR_MS_PER_DOC
    ratio_residue = ms_per_doc_residue / DETECTOR_MS_PER_DOC

    result = {
        "model": NLI_MODEL,
        "frame": "substitution",
        "n_spans": len(spans_meta), "n_scored": n_scored, "n_no_context": n_no_context,
        "n_prep_filtered": n_prep_filtered, "n_docs": n_docs,
        "auc": auc,
        "chosen_tau": chosen_tau,
        "tau_at_fdr_002": tau_002[0] if tau_002 else None,
        "tau_at_fdr_005": chosen_tau,
        "tau_at_fdr_010": tau_010[0] if tau_010 else None,
        "best_youden": best_youden,
        "operating_points": ops,
        "eval": {"keeps": len(keep_ents), "drops": len(drop_ents), "residue": n_residue},
        "rejection_precision": rejection_precision,
        "drop_recall": drop_recall,
        "keep_false_drop_rate": keep_false_drop_rate,
        "per_type_tau": per_type,
        "sweep": sweep(keep_ents, drop_ents),
        "action_counts_nli": act_counts("nli_action"),
        "action_counts_margin": dict(sorted(margin_counts.items())),
        "crosstab_margindrop": crosstab,
        "margin_dropped_but_v3_kept_n": len(margin_dropped_but_v3_kept),
        "v3_dropped_but_margin_kept_n": len(v3_dropped_but_margin_kept),
        "cost": {
            "ms_per_span": round(ms_per_span, 3),
            "spans_per_doc": round(spans_per_doc, 3),
            "ms_per_doc_all": round(ms_per_doc_all, 3),
            "ms_per_doc_residue": round(ms_per_doc_residue, 3),
            "ratio_all": round(ratio_all, 4),
            "ratio_residue": round(ratio_residue, 4),
            "residue_frac": round(residue_frac, 4),
            "detector_ms_per_doc": DETECTOR_MS_PER_DOC,
        },
        "examples": {
            "v3_dropped": v3_dropped,
            "margin_dropped_but_v3_kept": margin_dropped_but_v3_kept[:40],
        },
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2) + "\n")

    # stdout: AUC + operating-point table + chosen tau + cost ratios + margin crosstab (counts only)
    print(f"frame=substitution spans={len(spans_meta)} scored={n_scored} "
          f"no_context={n_no_context} prep_filtered={n_prep_filtered} docs={n_docs}")
    print(f"eval keeps={len(keep_ents)} drops={len(drop_ents)} residue={n_residue}")
    print(f"AUC(junk-vs-real)={auc}")
    print("operating points (max real-loss -> junk-dropped @ tau):")
    for op in ops:
        print(f"  <={op['max_real_loss']:.0%}: junk_dropped={op['junk_dropped']} "
              f"real_loss={op['real_loss']} tau={op['tau']}")
    print(f"best_youden={best_youden}")
    print(f"chosen_tau={chosen_tau} (fdr.02={result['tau_at_fdr_002']} "
          f"fdr.10={result['tau_at_fdr_010']})")
    print(f"rejection_precision={rejection_precision} drop_recall={drop_recall} "
          f"keep_false_drop_rate={keep_false_drop_rate}")
    print(f"crosstab margin-dropped -> nli: {json.dumps(crosstab)}")
    print(f"margin_dropped_but_v3_kept={len(margin_dropped_but_v3_kept)} "
          f"v3_dropped_but_margin_kept={len(v3_dropped_but_margin_kept)}")
    print(f"cost ms/span={ms_per_span:.2f} ms/doc(all)={ms_per_doc_all:.1f} "
          f"ratio_all={ratio_all:.2f} | residue_frac={residue_frac:.2f} "
          f"ms/doc(residue)={ms_per_doc_residue:.1f} ratio_residue={ratio_residue:.2f}")
    print(f"wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
