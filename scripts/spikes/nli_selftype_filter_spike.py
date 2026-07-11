"""Spike: NLI v2 binary SELF-TYPE verification as a detector-noise filter.

Throwaway probe (scripts/spikes/). v2 is deliberately simpler than the v1 contrastive spike
(nli_noise_filter_spike.py): for each detected span we form exactly ONE hypothesis --

    "<Surface> is <own-type phrase>."   (its OWN detected type; no distractors, no argmax, no retype)

premise = the span's containing sentence. KEEP iff entailment >= tau, else DROP. One NLI pair per
span. This does not retype; it only asks "is this span really what the detector called it?", which
catches deny-list junk and out-of-scope survivors cheaply.

The key improvement over v1 is a REPRESENTATIVE calibration drawn from the SAME real detected spans
(real sentences), not synthetic probes:
  - eval-KEEP = spans whose _norm surface hits its own type in the lattice profiles
    (lookup_entry) -- a known real entity of that type.
  - eval-DROP = spans whose _norm surface is in a detector deny-list junk set -- known junk.
tau is calibrated so eval-KEEP false-drop-rate stays low while eval-DROP recall is maximized.

Reuses the v1 NLI batching pattern (flatten pairs, chunk 128, read the "entailment" score) and its
doc_id->sentence loading (load_task_docs + sentence_of).

GPU: the real run loads the NLI model + clinical corpus. Run --smoke (model-free, corpus-free) for
the runnable check.

    PYTHONPATH=src:scripts .venv/bin/python scripts/spikes/nli_selftype_filter_spike.py --smoke
    PYTHONPATH=src:scripts .venv/bin/python scripts/spikes/nli_selftype_filter_spike.py         # real
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from cloak.lattice import NLI_MODEL

SPANS_PATH = Path("results/mined_lattice_profile_spans_large.jsonl")
OUT_PATH = Path("results/nli_selftype_spike.json")
DETECTOR_MS_PER_DOC = 324.0  # measured detector reference

# One self-type phrase per runtime type. A span whose type has no phrase here is skipped entirely.
TYPE_PHRASES = {
    "health-condition": "a disease, disorder, or medical condition",
    "medical-procedure": "a medical procedure, test, examination, or treatment",
    "drug": "a medication, drug, or dietary supplement",
    "injury": "a physical injury or trauma",
    "organization-medical-facility": "a hospital, clinic, or healthcare facility",
    "LOC": "a geographic place or location",
    "profession": "a job, occupation, or profession",
    "nationality": "a nationality",
    "religion": "a religion",
    "ethnicity": "an ethnicity",
}


def hypothesis(surface: str, phrase: str) -> str:
    return f"{surface.capitalize()} is {phrase}."


# ---------- scoring (scorer-injected; shared by real + smoke) ----------

def score_selftype(spans_meta: list[dict], scorer) -> None:
    """Attach `ent` (entailment of the span's single self-type hypothesis) to each span in place.
    Spans whose orig_type has no phrase get ent=None and are excluded from all downstream stats.
    `scorer` maps [(premise, hypothesis)] -> [entailment_prob]."""
    pairs: list[tuple[str, str]] = []
    idx: list[int] = []
    for i, m in enumerate(spans_meta):
        phrase = TYPE_PHRASES.get(m["orig_type"])
        if phrase is None:
            m["ent"] = None
            continue
        pairs.append((m["premise"], hypothesis(m["surface"], phrase)))
        idx.append(i)
    t0 = time.time()
    ents = scorer(pairs)
    elapsed_ms = (time.time() - t0) * 1000.0
    assert len(ents) == len(pairs), f"scorer returned {len(ents)} for {len(pairs)} pairs"
    for j, i in enumerate(idx):
        spans_meta[i]["ent"] = float(ents[j])
    spans_meta_meta = {"n_scored": len(pairs), "nli_ms": elapsed_ms}
    return spans_meta_meta  # noqa: RET504


# ---------- calibration ----------

def calibrate(keep_ents: list[float], drop_ents: list[float], fdr_bar: float):
    """Largest tau with eval-KEEP false-drop-rate <= fdr_bar (maximizes eval-DROP recall, which is
    monotone in tau). Returns (tau, drop_recall, keep_false_drop_rate) or None if infeasible."""
    if not keep_ents:
        return None
    cands = sorted(set(keep_ents + drop_ents))
    # also probe just-above each candidate so a tau can drop everything at-or-below it
    best = None
    for tau in cands:
        fdr = sum(e < tau for e in keep_ents) / len(keep_ents)
        if fdr <= fdr_bar:
            rec = (sum(e < tau for e in drop_ents) / len(drop_ents)) if drop_ents else 0.0
            if best is None or rec > best[1] or (rec == best[1] and tau > best[0]):
                best = (tau, rec, fdr)
    return best


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


# ---------- scorers ----------

def real_scorer(batch: int = 128):
    import torch
    from transformers import pipeline
    nli = pipeline("text-classification", model=NLI_MODEL,
                   device=0 if torch.cuda.is_available() else -1)

    def score(pairs: list[tuple[str, str]]) -> list[float]:
        out: list[float] = []
        for i in range(0, len(pairs), batch):
            chunk = pairs[i:i + batch]
            res = nli([{"text": p, "text_pair": h} for p, h in chunk], top_k=None, truncation=True)
            for scores in res:
                out.append(next(d["score"] for d in scores if d["label"] == "entailment"))
            print(f"nli {min(i + batch, len(pairs))}/{len(pairs)}", flush=True)
        return out

    return score


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


def margin_action_of(spans_meta: list[dict]) -> dict[tuple[str, str], str]:
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

# Synthetic spans carry an explicit eval label (real path derives it from lookup_entry/deny-list).
# Fake NLI: entailment keyed on the surface in the hypothesis. No model, no corpus.
SMOKE_SPANS = [
    # eval-KEEP: high entailment on own type
    {"surface": "blorbitis", "orig_type": "health-condition", "eval": "keep", "_ent": 0.95},
    {"surface": "ankleoid", "orig_type": "injury", "eval": "keep", "_ent": 0.88},
    {"surface": "zorbivex", "orig_type": "drug", "eval": "keep", "_ent": 0.90},
    {"surface": "clinoplace", "orig_type": "organization-medical-facility", "eval": "keep", "_ent": 0.85},
    {"surface": "scanwork", "orig_type": "medical-procedure", "eval": "keep", "_ent": 0.82},
    {"surface": "coastville", "orig_type": "LOC", "eval": "keep", "_ent": 0.80},
    # eval-DROP: junk, low entailment on the claimed type
    {"surface": "brickthing", "orig_type": "drug", "eval": "drop", "_ent": 0.10},
    {"surface": "woodplank", "orig_type": "medical-procedure", "eval": "drop", "_ent": 0.15},
    {"surface": "greyish", "orig_type": "health-condition", "eval": "drop", "_ent": 0.20},
    {"surface": "blankword", "orig_type": "injury", "eval": "drop", "_ent": 0.05},
    {"surface": "noiseblob", "orig_type": "organization-medical-facility", "eval": "drop", "_ent": 0.12},
    # unlabeled residue: only used in the applied run
    {"surface": "midword", "orig_type": "drug", "eval": None, "_ent": 0.55},
    {"surface": "otherword", "orig_type": "health-condition", "eval": None, "_ent": 0.40},
    # no-phrase type: skipped entirely (ent=None)
    {"surface": "someword", "orig_type": "gender", "eval": None, "_ent": 0.99},
]


def smoke_scorer_factory(spans_meta):
    ent_of = {}
    for m in spans_meta:
        ent_of[hypothesis(m["surface"], TYPE_PHRASES.get(m["orig_type"], "")).lower()] = m["_ent"]

    def score(pairs):
        return [ent_of.get(h.lower(), 0.1) for _p, h in pairs]

    return score


def run_smoke() -> int:
    spans_meta = [{"surface": m["surface"], "orig_type": m["orig_type"],
                   "premise": f"The chart says {m['surface']} was noted.", "eval": m["eval"],
                   "_ent": m["_ent"]}
                  for m in SMOKE_SPANS]
    score = smoke_scorer_factory(spans_meta)
    score_selftype(spans_meta, score)

    scored = [m for m in spans_meta if m["ent"] is not None]
    assert len(scored) == len(SMOKE_SPANS) - 1, "the no-phrase span should be skipped"
    keep_ents = [m["ent"] for m in scored if m["eval"] == "keep"]
    drop_ents = [m["ent"] for m in scored if m["eval"] == "drop"]

    chosen = calibrate(keep_ents, drop_ents, 0.05)
    assert chosen is not None, "calibration failed to pick a finite tau"
    tau = chosen[0]
    kept = [m for m in scored if m["ent"] >= tau]
    dropped = [m for m in scored if m["ent"] < tau]
    assert kept, "no spans kept at chosen tau"
    assert dropped, "no spans dropped at chosen tau"

    print(f"SMOKE scored={len(scored)} eval_keep={len(keep_ents)} eval_drop={len(drop_ents)}")
    print(f"SMOKE chosen_tau={tau:.3f} drop_recall={chosen[1]:.3f} keep_fdr={chosen[2]:.3f}")
    print(f"SMOKE applied: keep={len(kept)} drop={len(dropped)}")
    print("SMOKE OK: keep AND drop both occur; calibration picked a finite tau")
    return 0


# ---------- main (real) ----------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        return run_smoke()

    spans_meta, n_no_context, n_docs = load_real_spans()
    meta = score_selftype(spans_meta, real_scorer())
    scored = [m for m in spans_meta if m["ent"] is not None]
    n_scored, nli_ms = meta["n_scored"], meta["nli_ms"]

    keep_ents = [m["ent"] for m in scored if m["eval"] == "keep"]
    drop_ents = [m["ent"] for m in scored if m["eval"] == "drop"]
    n_residue = sum(1 for m in scored if m["eval"] is None)

    chosen = calibrate(keep_ents, drop_ents, 0.05)
    tau_002 = calibrate(keep_ents, drop_ents, 0.02)
    tau_010 = calibrate(keep_ents, drop_ents, 0.10)
    chosen_tau = chosen[0] if chosen else None

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
    action_of = {}
    for m in scored:
        m["nli_action"] = "keep" if (chosen_tau is not None and m["ent"] >= chosen_tau) else "drop"
        action_of[m["surface"] + "\x00" + m["orig_type"]] = m["nli_action"]

    margins, span_key = margin_action_of(scored)

    def m_act(m):
        return margins.get(span_key(m["surface"], m["orig_type"]), "none")

    crosstab = {"nli_keep": 0, "nli_drop": 0}
    margin_dropped_but_nli_kept = []
    nli_dropped_but_margin_kept = []
    for m in scored:
        ma = m_act(m)
        if ma == "drop-margin":
            crosstab["nli_" + m["nli_action"]] += 1
            if m["nli_action"] == "keep":
                margin_dropped_but_nli_kept.append(
                    {"surface": m["surface"], "type": m["orig_type"], "ent": round(m["ent"], 4)})
        if m["nli_action"] == "drop" and ma not in ("drop-margin", "drop-denylist", "none"):
            nli_dropped_but_margin_kept.append(
                {"surface": m["surface"], "type": m["orig_type"], "ent": round(m["ent"], 4)})

    nli_dropped = [{"surface": m["surface"], "type": m["orig_type"], "ent": round(m["ent"], 4)}
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

    # COST
    spans_per_doc = n_scored / n_docs if n_docs else 0.0
    ms_per_span = nli_ms / n_scored if n_scored else 0.0
    ms_per_doc_all = ms_per_span * spans_per_doc
    residue_frac = n_residue / n_scored if n_scored else 0.0
    ms_per_doc_residue = ms_per_doc_all * residue_frac
    ratio_all = ms_per_doc_all / DETECTOR_MS_PER_DOC
    ratio_residue = ms_per_doc_residue / DETECTOR_MS_PER_DOC

    result = {
        "model": NLI_MODEL,
        "n_spans": len(spans_meta), "n_scored": n_scored, "n_no_context": n_no_context,
        "n_docs": n_docs,
        "chosen_tau": chosen_tau,
        "tau_at_fdr_002": tau_002[0] if tau_002 else None,
        "tau_at_fdr_005": chosen_tau,
        "tau_at_fdr_010": tau_010[0] if tau_010 else None,
        "eval": {"keeps": len(keep_ents), "drops": len(drop_ents), "residue": n_residue},
        "rejection_precision": rejection_precision,
        "drop_recall": drop_recall,
        "keep_false_drop_rate": keep_false_drop_rate,
        "per_type_tau": per_type,
        "sweep": sweep(keep_ents, drop_ents),
        "action_counts_nli": act_counts("nli_action"),
        "action_counts_margin": dict(sorted(margin_counts.items())),
        "crosstab_margindrop": crosstab,
        "margin_dropped_but_nli_kept_n": len(margin_dropped_but_nli_kept),
        "nli_dropped_but_margin_kept_n": len(nli_dropped_but_margin_kept),
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
            "nli_dropped": nli_dropped,
            "margin_dropped_but_nli_kept": margin_dropped_but_nli_kept[:40],
        },
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2) + "\n")

    # stdout: counts + eval P/R + chosen_tau + cost ratios ONLY (no surface listings)
    print(f"spans={len(spans_meta)} scored={n_scored} no_context={n_no_context} docs={n_docs}")
    print(f"eval keeps={len(keep_ents)} drops={len(drop_ents)} residue={n_residue}")
    print(f"chosen_tau={chosen_tau} "
          f"(fdr.02={result['tau_at_fdr_002']} fdr.10={result['tau_at_fdr_010']})")
    print(f"rejection_precision={rejection_precision} drop_recall={drop_recall} "
          f"keep_false_drop_rate={keep_false_drop_rate}")
    print(f"crosstab margin-dropped -> nli: {json.dumps(crosstab)}")
    print(f"margin_dropped_but_nli_kept={len(margin_dropped_but_nli_kept)} "
          f"nli_dropped_but_margin_kept={len(nli_dropped_but_margin_kept)}")
    print(f"cost ms/span={ms_per_span:.2f} ms/doc(all)={ms_per_doc_all:.1f} "
          f"ratio_all={ratio_all:.2f} | residue_frac={residue_frac:.2f} "
          f"ms/doc(residue)={ms_per_doc_residue:.1f} ratio_residue={ratio_residue:.2f}")
    print(f"wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
