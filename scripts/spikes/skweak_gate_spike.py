"""Skweak adoption spike: does fused weak supervision beat the fixed layer order?

Spec: docs/specs/detector-noise-semantic-gate.md, "Aggregation (skweak) — fitted offline,
applied everywhere" + the adoption rule. Question: fit skweak's label model over the span
gate's layer outputs (as labelling functions) and compare keep/drop quality against the plain
layered gate (fixed layer precedence, frozen calibration) on the labeled eval set. Adopt only
if fused beats layered on **drop-precision at >= equal drop-recall**.

skweak is span-sequence oriented; we use its **text-classification** aggregator
(`skweak.generative.NaiveBayes`, a `TextAggregatorMixin`) and model each span as a 1-token
spaCy doc with a single (0,1) span. This is the genuine skweak API — NaiveBayes fits a
generative label model over the per-source vote matrix and returns a keep/drop posterior per
doc — so no HMM-free fallback is needed. The output records the adaptation.

LFs over each unique (surface, runtime_type), voting keep/drop or abstaining:
  - denylist : cloak.detect.is_noise_span -> drop | abstain
  - link     : lattice_profiles.lookup_entry own/other profile-backed type -> keep | abstain
  - margin   : pos/neg cosine vs the real embindex + negatives.npz, frozen miner floor/margin
               -> drop (pos<floor & neg-pos>=margin) | keep (pos>=floor) | abstain
  - score    : detector score <0.5 -> drop | 0.5-0.8 abstain | >0.8 -> keep (abstains w/o score)

Eval set (== calibrate_span_gate.py construction): keeps = profile surfaces of the
noise-filter types; drops = eval half of anchor_seed_split(seed_negative_surfaces()).

All embedding happens in ONE batched model.encode (a per-surface cache warmed up front and
shared with gate_spans). --smoke runs the whole path on 20 synthetic spans with a fake
embed_fn and no model / file loads.

  PYTHONPATH=src:scripts .venv/bin/python -u scripts/spikes/skweak_gate_spike.py
  PYTHONPATH=src:scripts .venv/bin/python -u scripts/spikes/skweak_gate_spike.py --smoke
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import spacy
from spacy.tokens import Doc, Span
from skweak.generative import NaiveBayes

from cloak import lattice_profiles as lp
from cloak.detect import _NOISE_FILTER_TYPES, is_noise_span
from cloak.profile_match import (PROFILE_BACKED_TYPES, _index_path_for, _l2norm, _st_model,
                                 load_embindex, span_key)
from cloak.span_gate import (DEFAULT_CALIBRATION_PATH, DEFAULT_NEGATIVES_PATH, _load_negatives,
                             anchor_seed_split, gate_spans, load_calibration,
                             seed_negative_surfaces)

SPANS = ROOT / "results/mined_lattice_profile_spans_large.jsonl"
OUT = ROOT / "results/skweak_spike.json"
LABELS = ["keep", "drop"]
LF_NAMES = ("denylist", "link", "margin", "score")


# --- labelling functions ---------------------------------------------------------------
# Each returns "keep" / "drop" / None (abstain). Pure LFs (denylist) call cloak directly;
# the rest take injected primitives so --smoke needs no file/model loads.

def lf_denylist(surface, rt):
    return "drop" if is_noise_span(surface, rt) else None


def lf_link(surface, rt, link_fn):
    """link_fn(surface, rt) -> True iff surface resolves to a profile entry of a
    profile-backed type (own type or any other) => keep."""
    return "keep" if link_fn(surface, rt) else None


def lf_margin(vec, rt, pos_sim_fn, neg_vectors, floor, margin):
    if vec is None or neg_vectors.size == 0:
        return None
    pos = pos_sim_fn(rt, vec)
    if pos is None:
        return None
    neg = float(np.max(neg_vectors @ vec))
    if pos < floor and (neg - pos) >= margin:
        return "drop"
    if pos >= floor:
        return "keep"
    return None


def lf_score(score):
    if score is None:
        return None
    if score < 0.5:
        return "drop"
    if score > 0.8:
        return "keep"
    return None


def votes_for(surface, rt, score, vec, *, link_fn, pos_sim_fn, neg_vectors, floor, margin):
    return {
        "denylist": lf_denylist(surface, rt),
        "link": lf_link(surface, rt, link_fn),
        "margin": lf_margin(vec, rt, pos_sim_fn, neg_vectors, floor, margin),
        "score": lf_score(score),
    }


# --- skweak plumbing -------------------------------------------------------------------

def make_doc(vocab, surface, votes):
    """One span == one 1-token doc; each firing LF adds a (0,1) span with its keep/drop label."""
    doc = Doc(vocab, words=[surface if surface else "∅"])
    for lf in LF_NAMES:
        label = votes.get(lf)
        doc.spans[lf] = [Span(doc, 0, 1, label=label)] if label is not None else []
    return doc


def fused_drop_probs(fit_votes, eval_votes, vocab):
    """Fit NaiveBayes on the mined-corpus votes, return P(drop) for each eval doc."""
    fit_docs = [make_doc(vocab, s, v) for s, v in fit_votes]
    model = NaiveBayes("skweak_gate", LABELS)
    model.fit(fit_docs)
    probs = []
    for s, v in eval_votes:
        doc = model(make_doc(vocab, s, v))
        agg = doc.spans[model.name].attrs.get("probs", {})
        probs.append(agg.get((0, 1), {}).get("drop", 0.0))
    return np.asarray(probs, dtype=float)


# --- metrics ---------------------------------------------------------------------------

def score_dropmask(is_drop_truth, dropped):
    """is_drop_truth, dropped: bool arrays over the eval items (keeps then drops order-agnostic)."""
    is_drop_truth = np.asarray(is_drop_truth, dtype=bool)
    dropped = np.asarray(dropped, dtype=bool)
    n_drop_truth = int(is_drop_truth.sum())
    n_keep_truth = int((~is_drop_truth).sum())
    tp = int((dropped & is_drop_truth).sum())          # true drops dropped
    fp = int((dropped & ~is_drop_truth).sum())         # keeps wrongly dropped
    total_dropped = tp + fp
    return {
        "drop_precision": tp / total_dropped if total_dropped else 0.0,
        "drop_recall": tp / n_drop_truth if n_drop_truth else 0.0,
        "false_drop_rate": fp / n_keep_truth if n_keep_truth else 0.0,
        "n_dropped": total_dropped,
    }


def pick_fused_operating_point(probs, is_drop_truth, target_recall):
    """Lowest posterior threshold whose drop_recall >= target_recall, max precision among ties.

    Returns (threshold, metrics). Falls back to the max-recall threshold if the target is
    unreachable (fused then cannot match the layered gate)."""
    cands = sorted(set(np.round(probs, 4)) | {0.5})
    best = None
    for thr in cands:
        m = score_dropmask(is_drop_truth, probs > thr)
        m = {**m, "threshold": float(thr)}
        if m["drop_recall"] >= target_recall:
            key = (m["drop_precision"], -thr)
            if best is None or key > best[0]:
                best = (key, m)
    if best is not None:
        return best[1]["threshold"], best[1]
    # unreachable: report the threshold giving the highest recall (smallest thr)
    m = score_dropmask(is_drop_truth, probs > cands[0])
    return float(cands[0]), {**m, "threshold": float(cands[0])}


# --- real inputs -----------------------------------------------------------------------

def build_real_inputs():
    from build_mined_lattice_profiles import normalize_detector_label

    profiles_path = lp.DEFAULT_PROFILE_PATH
    index = load_embindex(str(_index_path_for(profiles_path)), str(profiles_path))
    if index is None:
        raise SystemExit(f"no usable embindex for {profiles_path} (build it first)")
    negatives = _load_negatives(DEFAULT_NEGATIVES_PATH)
    if negatives is None:
        raise SystemExit(f"no negatives index at {DEFAULT_NEGATIVES_PATH}")
    neg_vectors, _ = negatives
    calibration = load_calibration(DEFAULT_CALIBRATION_PATH) or {}
    point = (calibration.get("points") or {}).get("miner")
    if point is None:
        raise SystemExit("no miner calibration point (calibrate the gate first)")
    floor, margin = float(point["floor"]), float(point["margin"])

    # fit corpus: mined re-mine spans -> (surface, runtime_type, score)
    rows = [json.loads(l) for l in SPANS.read_text().splitlines() if l.strip()]
    fit_spans = [(r["surface"], normalize_detector_label(r["detector_label"]), float(r["score"]))
                 for r in rows]

    # eval set (== calibrate_span_gate.py)
    artifact = lp.load_profiles(profiles_path)
    keep_items = sorted({(s, rt) for rt in _NOISE_FILTER_TYPES
                         for canon, row in artifact.get("profiles", {}).get(rt, {}).items()
                         for s in (canon, *row.get("aliases", []))})
    _, eval_half = anchor_seed_split(seed_negative_surfaces())
    # negatives carry no inherent runtime_type; assign a noise-filter type so is_noise_span /
    # per-type margin fire, matching how the miner would see a mislabeled clinical noise span.
    drop_rt = "health-condition"
    drop_items = [(s, drop_rt) for s in eval_half]

    # one batched embedding for every surface we will score (fit + eval), shared with gate_spans
    all_surfaces = sorted({s for s, _, _ in fit_spans}
                          | {s for s, _ in keep_items} | {s for s, _ in drop_items})
    model = _st_model(index.model_id)
    vecs = _l2norm(np.asarray(model.encode(all_surfaces, normalize_embeddings=True),
                              dtype=np.float32))
    cache = {s: v for s, v in zip(all_surfaces, vecs)}

    def cached_embed(texts):
        missing = [t for t in texts if t not in cache]
        if missing:
            mv = _l2norm(np.asarray(model.encode(missing, normalize_embeddings=True),
                                    dtype=np.float32))
            cache.update(zip(missing, mv))
        return np.asarray([cache[t] for t in texts], dtype=np.float32)

    def link_fn(surface, rt):
        if lp.lookup_entry(surface, rt, profiles_path):
            return True
        return any(lp.lookup_entry(surface, other, profiles_path)
                   for other in PROFILE_BACKED_TYPES if other != rt)

    def pos_sim_fn(rt, vec):
        r = index.type_rows(rt)
        if not r:
            return None
        return float(np.max(index.vectors[r] @ vec))

    return dict(fit_spans=fit_spans, keep_items=keep_items, drop_items=drop_items,
                cache=cache, cached_embed=cached_embed, link_fn=link_fn, pos_sim_fn=pos_sim_fn,
                neg_vectors=neg_vectors, floor=floor, margin=margin, profiles_path=profiles_path,
                model_id=index.model_id)


# --- synthetic inputs (--smoke): no model, no files ------------------------------------

def build_smoke_inputs():
    rng = np.random.default_rng(0)
    dim = 8

    def fake_vec(text):
        r = np.random.default_rng(abs(hash(text)) % (2 ** 32))
        return _l2norm(r.standard_normal((1, dim)).astype(np.float32))[0]

    # 10 keep-ish + 10 drop-ish synthetic mined spans. Drop surfaces contain "panel" so the
    # denylist LF (is_noise_span, real regex) fires deterministically -> exercises the drop path.
    fit_spans = []
    for i in range(10):
        fit_spans.append((f"keepterm{i}", "drug", 0.95))
        fit_spans.append((f"lab panel {i}", "health-condition", 0.2))
    keep_items = [(f"keepterm{i}", "drug") for i in range(6)]
    drop_items = [(f"lab panel {i}", "health-condition") for i in range(6)]

    all_surfaces = {s for s, _, _ in fit_spans} | {s for s, _ in keep_items + drop_items}
    cache = {s: fake_vec(s) for s in all_surfaces}
    neg_vectors = _l2norm(rng.standard_normal((5, dim)).astype(np.float32))
    # a fake positive anchor per keep surface, so keepterms score high pos-sim
    pos_bank = {"drug": np.stack([cache[f"keepterm{i}"] for i in range(10)])}

    def link_fn(surface, rt):
        return surface.startswith("keepterm")  # fake profile membership

    def pos_sim_fn(rt, vec):
        bank = pos_bank.get(rt)
        return float(np.max(bank @ vec)) if bank is not None else None

    def cached_embed(texts):
        for t in texts:
            cache.setdefault(t, fake_vec(t))
        return np.asarray([cache[t] for t in texts], dtype=np.float32)

    return dict(fit_spans=fit_spans, keep_items=keep_items, drop_items=drop_items,
                cache=cache, cached_embed=cached_embed, link_fn=link_fn, pos_sim_fn=pos_sim_fn,
                neg_vectors=neg_vectors, floor=0.8, margin=0.0, profiles_path=None,
                model_id="fake")


# --- driver ----------------------------------------------------------------------------

def run(inp, *, smoke):
    vocab = spacy.blank("en").vocab

    # fit-corpus votes (all LFs active; scores present)
    fit_votes = []
    for s, rt, sc in inp["fit_spans"]:
        v = votes_for(s, rt, sc, inp["cache"].get(s), link_fn=inp["link_fn"],
                      pos_sim_fn=inp["pos_sim_fn"], neg_vectors=inp["neg_vectors"],
                      floor=inp["floor"], margin=inp["margin"])
        fit_votes.append((s, v))

    # eval votes (no detector score -> score LF abstains, as in production on profile/negatives)
    eval_items = [(s, rt, False) for s, rt in inp["keep_items"]] + \
                 [(s, rt, True) for s, rt in inp["drop_items"]]
    eval_votes, is_drop_truth = [], []
    for s, rt, truth in eval_items:
        v = votes_for(s, rt, None, inp["cache"].get(s), link_fn=inp["link_fn"],
                      pos_sim_fn=inp["pos_sim_fn"], neg_vectors=inp["neg_vectors"],
                      floor=inp["floor"], margin=inp["margin"])
        eval_votes.append((s, v))
        is_drop_truth.append(truth)
    is_drop_truth = np.asarray(is_drop_truth, dtype=bool)

    # (b) plain layered gate via gate_spans with the frozen calibration
    if smoke:
        # skip gate_spans (needs real profile/calibration artifacts); emulate its precedence
        # from the same LF primitives so the comparison path still runs end to end.
        layered_dropped = []
        for s, rt, _ in eval_items:
            if inp["link_fn"](s, rt):
                layered_dropped.append(False)
            elif is_noise_span(s, rt):
                layered_dropped.append(True)
            else:
                m = lf_margin(inp["cache"].get(s), rt, inp["pos_sim_fn"], inp["neg_vectors"],
                              inp["floor"], inp["margin"])
                layered_dropped.append(m == "drop")
        layered_dropped = np.asarray(layered_dropped, dtype=bool)
    else:
        items = [(s, rt) for s, rt, _ in eval_items]
        decisions = gate_spans(items, "miner", profiles_path=inp["profiles_path"],
                               embed_fn=inp["cached_embed"])
        layered_dropped = np.asarray(
            [decisions[span_key(s, rt)].action == "drop" for s, rt, _ in eval_items], dtype=bool)

    layered = score_dropmask(is_drop_truth, layered_dropped)

    # (a) fused weak-supervision model
    probs = fused_drop_probs(fit_votes, eval_votes, vocab)
    fused_thr, fused = pick_fused_operating_point(probs, is_drop_truth, layered["drop_recall"])

    # adoption rule: adopt only if fused beats layered on drop-precision at >= equal drop-recall
    adopt = (fused["drop_recall"] >= layered["drop_recall"]
             and fused["drop_precision"] > layered["drop_precision"])

    report = {
        "smoke": smoke,
        "method": ("skweak.generative.NaiveBayes (text-classification aggregator) over "
                   "1-token spaCy docs; genuine skweak API, no HMM-free fallback"),
        "model_id": inp["model_id"],
        "calibration": {"operating_point": "miner", "floor": inp["floor"], "margin": inp["margin"]},
        "eval": {"keeps": int((~is_drop_truth).sum()), "drops": int(is_drop_truth.sum()),
                 "fit_spans": len(fit_votes)},
        "layered": layered,
        "fused": {**fused, "posterior_threshold": fused_thr,
                  "recall_matched_to_layered": bool(fused["drop_recall"] >= layered["drop_recall"])},
        "verdict": {"adopt_skweak": bool(adopt),
                    "rule": "adopt iff fused drop_precision > layered AND fused drop_recall >= layered",
                    "reason": ("fused beats layered precision at >= recall" if adopt else
                               "fused does not beat layered precision at matched recall")},
    }
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true",
                    help="run the full path on 20 synthetic spans with a fake embed_fn; no model/file loads")
    args = ap.parse_args()

    inp = build_smoke_inputs() if args.smoke else build_real_inputs()
    report = run(inp, smoke=args.smoke)

    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # counts-only stdout (no term listings)
    e, lay, fu, ver = report["eval"], report["layered"], report["fused"], report["verdict"]
    print(f"{'SMOKE ' if args.smoke else ''}eval: keeps={e['keeps']} drops={e['drops']} "
          f"fit_spans={e['fit_spans']}")
    print(f"layered: drop_precision={lay['drop_precision']:.4f} drop_recall={lay['drop_recall']:.4f} "
          f"false_drop_rate={lay['false_drop_rate']:.4f} n_dropped={lay['n_dropped']}")
    print(f"fused  : drop_precision={fu['drop_precision']:.4f} drop_recall={fu['drop_recall']:.4f} "
          f"false_drop_rate={fu['false_drop_rate']:.4f} n_dropped={fu['n_dropped']} "
          f"thr={fu['posterior_threshold']:.4f}")
    print(f"verdict: adopt_skweak={ver['adopt_skweak']} ({ver['reason']})")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
