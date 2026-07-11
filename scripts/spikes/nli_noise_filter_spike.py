"""Spike: NLI type-verification as a detector-noise filter vs the span_gate margin layer.

Throwaway probe (scripts/spikes/). Question: on the cached re-mine spans, does entailment over
contrastive type hypotheses (all target types + distractor "none-of-target" phrases, argmax
reroute) keep the REAL entities the margin layer wrongly drops -- and does it drop genuine noise?

Head-to-head: for spans the margin layer dropped, what does NLI do (does it recover them)?
For spans NLI drops, what did the margin layer do?

Literature the design leans on:
  - SATORI (arXiv 2401.16293): per-type thresholding, contrastive verbalization.
  - SummaC (arXiv 2111.09525): sentence-level NLI as a factual-consistency gate.
Lessons applied: per-type target phrases, distractor 'none-of-target' hypotheses, argmax over
all candidate types (target reroute), one tunable threshold TAU.

Reuses cloak.lattice.NLI_MODEL and the HF text-classification pipeline pattern (NOT
nli_gate_batch -- that substitutes candidates INTO the sentence; wrong shape here). We ask,
per span, "<surface> is <type phrase>." entailed by its containing sentence.

GPU: the real run loads the NLI model. Run --smoke (model-free, corpus-free) for the check.

    PYTHONPATH=src:scripts .venv/bin/python scripts/spikes/nli_noise_filter_spike.py --smoke
    PYTHONPATH=src:scripts .venv/bin/python scripts/spikes/nli_noise_filter_spike.py         # real
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cloak.lattice import NLI_MODEL

SPANS_PATH = Path("results/mined_lattice_profile_spans_large.jsonl")
OUT_PATH = Path("results/nli_noise_filter_spike.json")

# Contrastive verbalizations. Target types are what a survivor may be (KEEP or RETYPE); a span
# whose best entailment lands on a distractor is out-of-scope noise (DROP). The clinical miner
# only emits the medical target types; LOC/profession are reroute-only destinations.
TARGET_TYPE_PHRASES = {
    "health-condition": "a medical condition or disease",
    "medical-procedure": "a medical procedure, test, or treatment",
    "drug": "a medication, drug, or supplement",
    "injury": "a physical injury",
    "LOC": "a geographic location",
    "organization-medical-facility": "a hospital, clinic, or medical facility",
    "profession": "an occupation or profession",
}
DISTRACTOR_PHRASES = {
    "__body_part": "a body part or anatomical structure",
    "__object": "a physical object or material",
    "__activity": "an everyday activity or action",
    "__attribute": "a generic attribute or state",
}
ALL_PHRASES = {**TARGET_TYPE_PHRASES, **DISTRACTOR_PHRASES}


def hypothesis(surface: str, phrase: str) -> str:
    return f"{surface.capitalize()} is {phrase}."


# ---------- core decision loop (scorer-injected; shared by real + smoke) ----------

def run_nli_filter(spans_meta: list[dict], scorer, tau: float) -> list[dict]:
    """spans_meta rows: {surface, orig_type, premise}. `scorer` maps [(premise, hypothesis)] ->
    [entailment_prob]. Returns a decision row per span."""
    type_keys = list(ALL_PHRASES)  # stable order per span
    pairs: list[tuple[str, str]] = []
    for m in spans_meta:
        for tk in type_keys:
            pairs.append((m["premise"], hypothesis(m["surface"], ALL_PHRASES[tk])))
    ents = scorer(pairs)
    assert len(ents) == len(pairs), f"scorer returned {len(ents)} for {len(pairs)} pairs"

    rows: list[dict] = []
    n = len(type_keys)
    for i, m in enumerate(spans_meta):
        block = ents[i * n:(i + 1) * n]
        by_type = dict(zip(type_keys, block))
        best_type = max(by_type, key=by_type.get)
        best_ent = float(by_type[best_type])
        own_ent = float(by_type.get(m["orig_type"], 0.0))  # detector type's own entailment
        if best_ent < tau:
            action, reason = "drop", "low-entailment"
        elif best_type in DISTRACTOR_PHRASES:
            action, reason = "drop", best_type
        elif best_type == m["orig_type"]:
            action, reason = "keep", "type-confirmed"
        else:
            action, reason = "retype", best_type
        rows.append({
            "surface": m["surface"], "orig_type": m["orig_type"],
            "best_type": best_type, "best_ent": round(best_ent, 4),
            "own_ent": round(own_ent, 4), "action": action, "reason": reason,
        })
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


# Deterministic fake for --smoke: entailment high iff a surface's truth keyword is present in the
# hypothesis phrase. No model, no corpus. Keyed by surface.
SMOKE_TRUTH = {
    # KEEP: best entailment on the detector's own type
    "blorbectomy": "procedure",       # medical-procedure -> keep
    "zorbitol": "medication",         # drug -> keep
    "frangitis": "disease",           # health-condition -> keep
    "clinovex center": "clinic",      # organization-medical-facility -> keep
    # RETYPE: best entailment on a different target type
    "ankleoid": "injury",             # detected health-condition -> retype injury
    "plovaquine": "medication",       # detected health-condition -> retype drug
    "sporeton": "procedure",          # detected drug -> retype medical-procedure
    # DROP (distractor argmax): out-of-scope
    "brickthing": "object",           # -> __object
    "wibblearm": "body",              # -> __body_part
    "frobnicate": "activity",         # -> __activity
    "greenish": "attribute",          # -> __attribute
    # DROP (low-entailment): no keyword match anywhere
    "quxnonsense": None,
    "zzblank": None,
    "mumbleword": None,
    "flibberish": None,
}


def smoke_scorer(pairs: list[tuple[str, str]]) -> list[float]:
    out = []
    for _premise, hyp in pairs:
        hl = hyp.lower()
        kw = next((k for s, k in SMOKE_TRUTH.items() if hl.startswith(s + " is ")), None)
        out.append(0.9 if (kw and kw in hl) else 0.1)
    return out


def smoke_spans() -> list[dict]:
    # detector types are runtime types; sentences are generic, no clinical terms invented.
    orig = {
        "blorbectomy": "medical-procedure", "zorbitol": "drug", "frangitis": "health-condition",
        "clinovex center": "organization-medical-facility",
        "ankleoid": "health-condition", "plovaquine": "health-condition", "sporeton": "drug",
        "brickthing": "drug", "wibblearm": "injury", "frobnicate": "medical-procedure",
        "greenish": "health-condition",
        "quxnonsense": "drug", "zzblank": "health-condition", "mumbleword": "injury",
        "flibberish": "drug",
    }
    return [{"surface": s, "orig_type": t,
             "premise": f"The report noted that {s} was present in the record."}
            for s, t in orig.items()]


# ---------- real corpus wiring ----------

def load_real_spans() -> tuple[list[dict], int]:
    """Returns (spans_meta, n_no_context). Uses _unique_spans + normalize_detector_label +
    sentence_of; drops spans whose doc/sentence can't be located."""
    from build_mined_lattice_profiles import DetectedSpan, _unique_spans, normalize_detector_label
    from cloak.corpora import load_task_docs
    from cloak.train.ladder_probes import sentence_of

    raw = [json.loads(line) for line in SPANS_PATH.read_text().splitlines() if line.strip()]
    spans = [DetectedSpan(r["surface"], r["detector_label"], r["doc_id"], float(r["score"]))
             for r in raw]
    unique = _unique_spans(spans)

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
        meta.append({"surface": sp.surface,
                     "orig_type": normalize_detector_label(sp.detector_label),
                     "premise": premise})
    return meta, n_no_context


def margin_decisions(spans_meta: list[dict]) -> dict[tuple[str, str], object]:
    from cloak.profile_match import span_key
    from cloak.span_gate import gate_spans
    items = [(m["surface"], m["orig_type"]) for m in spans_meta]
    decisions = gate_spans(items, "miner")
    return {span_key(m["surface"], m["orig_type"]): decisions.get(span_key(m["surface"], m["orig_type"]))
            for m in spans_meta}, span_key


# ---------- reporting ----------

def counts(rows: list[dict], key: str) -> dict:
    out: dict[str, int] = {}
    for r in rows:
        out[r[key]] = out.get(r[key], 0) + 1
    return dict(sorted(out.items()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        spans_meta = smoke_spans()
        rows = run_nli_filter(spans_meta, smoke_scorer, args.tau)
        acts = counts(rows, "action")
        print("SMOKE nli action_counts:", json.dumps(acts))
        # crosstab-free smoke: just prove all three actions fire
        assert {"keep", "drop", "retype"} <= set(acts), f"missing actions: {acts}"
        assert any(r["reason"] == "low-entailment" for r in rows), "no low-entailment drop"
        assert any(r["reason"] in DISTRACTOR_PHRASES for r in rows), "no distractor drop"
        print("SMOKE OK: keep/drop/retype all produced; low-entailment + distractor drops present")
        return 0

    spans_meta, n_no_context = load_real_spans()
    print(f"spans={len(spans_meta)} no_context={n_no_context}", flush=True)
    rows = run_nli_filter(spans_meta, real_scorer(), args.tau)

    # head-to-head vs the margin layer
    margins, span_key = margin_decisions(spans_meta)
    margin_action_of: dict[str, str] = {}
    for m in spans_meta:
        d = margins.get(span_key(m["surface"], m["orig_type"]))
        if d is None:
            margin_action_of[m["surface"] + "\x00" + m["orig_type"]] = "none"
        elif d.action == "drop":
            margin_action_of[m["surface"] + "\x00" + m["orig_type"]] = "drop-" + d.layer  # drop-margin / drop-denylist
        else:
            margin_action_of[m["surface"] + "\x00" + m["orig_type"]] = d.action  # keep / retype

    def m_act(r):
        return margin_action_of.get(r["surface"] + "\x00" + r["orig_type"], "none")

    # KEY question: spans the margin layer dropped -- what does NLI do?
    crosstab = {"nli_keep": 0, "nli_drop": 0, "nli_retype": 0}
    margin_dropped_but_nli_kept = []
    for r in rows:
        if m_act(r) == "drop-margin":
            crosstab["nli_" + r["action"]] += 1
            if r["action"] in ("keep", "retype"):
                margin_dropped_but_nli_kept.append(
                    {"surface": r["surface"], "orig_type": r["orig_type"],
                     "best_type": r["best_type"], "best_ent": r["best_ent"]})

    # spans NLI drops -- what did the margin layer do?
    nli_drop_vs_margin: dict[str, int] = {}
    for r in rows:
        if r["action"] == "drop":
            a = m_act(r)
            nli_drop_vs_margin[a] = nli_drop_vs_margin.get(a, 0) + 1

    nli_dropped = [{"surface": r["surface"], "orig_type": r["orig_type"],
                    "best_type": r["best_type"], "best_ent": r["best_ent"]}
                   for r in rows if r["action"] == "drop"][:40]
    nli_retyped = [{"surface": r["surface"], "orig_type": r["orig_type"],
                    "best_type": r["best_type"], "best_ent": r["best_ent"]}
                   for r in rows if r["action"] == "retype"][:40]

    result = {
        "model": NLI_MODEL,
        "tau": args.tau,
        "n_spans": len(spans_meta),
        "n_no_context": n_no_context,
        "action_counts_nli": counts(rows, "action"),
        "action_counts_margin": dict(sorted(
            {v: sum(1 for x in margin_action_of.values() if x == v)
             for v in set(margin_action_of.values())}.items())),
        "crosstab_margindrop_vs_nli": crosstab,
        "nli_drop_vs_margin": dict(sorted(nli_drop_vs_margin.items())),
        "nli_dropped": nli_dropped,
        "nli_retyped": nli_retyped,
        "margin_dropped_but_nli_kept": margin_dropped_but_nli_kept[:40],
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2) + "\n")

    # stdout: counts + crosstab only, NO surface listings
    print("nli   action_counts:", json.dumps(result["action_counts_nli"]))
    print("margin action_counts:", json.dumps(result["action_counts_margin"]))
    print("crosstab (margin DROPPED -> nli):", json.dumps(crosstab))
    print("nli DROPPED -> margin action:", json.dumps(result["nli_drop_vs_margin"]))
    print(f"margin_dropped_but_nli_kept (headline recovery): {len(margin_dropped_but_nli_kept)}")
    print(f"wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
