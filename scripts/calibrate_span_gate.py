"""Calibrate the semantic span gate's anchor-margin layer at frozen operating points.

Builds an eval set from the profile artifact (keeps = canonical+alias surfaces of the
noise-filter types) and the held-out eval half of the deny-list negatives (drops =
anchor_seed_split eval half). Scores raw margin decisions on embeddings only (layer 1/2
bypassed), sweeps floor x margin, and freezes two operating points per the empirical-honesty
bars:
  - production: largest drop-recall with false-drop rate <= production_false_drop on keeps;
  - miner: largest drop-recall with drop-precision >= miner_precision first.
A point whose bar is unreachable is omitted -> gate_spans fail-opens (margin layer disabled).

Writes results/span_gate_calibration.json. Prints counts/rates only (never term listings).

Spec: docs/specs/detector-noise-semantic-gate.md.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from cloak import lattice_profiles as lp
from cloak.detect import _NOISE_FILTER_TYPES
from cloak.profile_match import (_index_path_for, _l2norm, _st_model, load_embindex)
from cloak.span_gate import (DEFAULT_CALIBRATION_PATH, DEFAULT_NEGATIVES_PATH, SCHEMA_VERSION,
                             _load_negatives, anchor_seed_split, build_negative_index,
                             seed_negative_surfaces)


def margin_scores(surfaces, index_vectors, neg_vectors, embed_fn) -> list[tuple[float, float]]:
    """For each surface: (pos, neg) = max cosine to the given positive rows / negative index.

    Primitive scored against a FIXED positive-row set. The eval builders below call it per
    runtime type (index.type_rows(t)) so pos matches gate_spans, which scores each span
    against its OWN type's rows only.
    """
    if len(surfaces) == 0:
        return []
    q = _l2norm(np.asarray(embed_fn(surfaces), dtype=np.float32))
    idx = np.asarray(index_vectors, dtype=np.float32)
    neg = np.asarray(neg_vectors, dtype=np.float32)
    out: list[tuple[float, float]] = []
    for v in q:
        pos = float(np.max(idx @ v)) if idx.size else 0.0
        nsim = float(np.max(neg @ v)) if neg.size else 0.0
        out.append((pos, nsim))
    return out


def keep_scores_per_type(keep_pairs, index, neg_vectors, embed_fn) -> list[tuple[float, float]]:
    """Score (surface, runtime_type) keeps EXACTLY as gate_spans: pos = max cosine to that
    surface's OWN type rows only (index.type_rows), never the pooled noise-filter anchors."""
    by_type: dict[str, list[str]] = {}
    for surface, rt in keep_pairs:
        by_type.setdefault(rt, []).append(surface)
    out: list[tuple[float, float]] = []
    for rt, surfaces in by_type.items():
        rows = index.type_rows(rt)
        out.extend(margin_scores(surfaces, index.vectors[rows], neg_vectors, embed_fn))
    return out


def drop_scores_worst_case(surfaces, index, neg_vectors, embed_fn, types
                           ) -> list[tuple[float, float]]:
    """Score eval drop-surfaces per noise-filter type and keep the WORST CASE across types.

    An eval negative arrives at runtime tagged as some noise-filter type, but which one is
    unknown here — so we score it against every type and take the configuration MOST LIKELY
    TO DROP: the lowest pos (a drop needs pos < floor AND neg-pos >= margin, both eased by a
    smaller pos; neg is type-independent). This makes drop_recall an optimistic bound — it
    credits a drop when ANY plausible type would trigger it — matching how the gate can fire
    under the span's actual tag. Types with no anchors fail-open (keep) in gate_spans, so
    they can never be the dropping config and are excluded from the min.
    """
    if not surfaces:
        return []
    typed = [(t, index.type_rows(t)) for t in types]
    typed = [(t, r) for t, r in typed if r]
    if not typed:   # no anchors for any noise-filter type -> gate fail-opens (keep)
        return [(1.0, 0.0)] * len(surfaces)
    per_type = [margin_scores(surfaces, index.vectors[r], neg_vectors, embed_fn)
                for _, r in typed]
    return [(min(pt[i][0] for pt in per_type), per_type[0][i][1])   # neg identical across types
            for i in range(len(surfaces))]


def _variants(surface: str) -> set[str]:
    """Cheap near-miss surface variants a robust gate must still keep: plural, article
    prefix, and (for surfaces > 6 chars) a one-char deletion typo."""
    vs = {surface + "s", "the " + surface}
    if len(surface) > 6:
        vs.add(surface[:1] + surface[2:])   # delete the 2nd char
    return vs


def _rates(keeps, drops, floor: float, margin: float) -> tuple[float, float, float]:
    def dropped(scores) -> int:
        return sum(1 for pos, neg in scores if pos < floor and (neg - pos) >= margin)

    keeps_dropped = dropped(keeps)
    drops_dropped = dropped(drops)
    false_drop_rate = keeps_dropped / len(keeps) if keeps else 0.0
    drop_recall = drops_dropped / len(drops) if drops else 0.0
    total_dropped = keeps_dropped + drops_dropped
    drop_precision = drops_dropped / total_dropped if total_dropped else 0.0
    return false_drop_rate, drop_recall, drop_precision


def choose_points(keeps, drops, *, floors, margins, production_false_drop, miner_precision
                  ) -> tuple[list[dict], dict]:
    """Sweep floor x margin, freeze the two operating points per the empirical-honesty bars."""
    sweep: list[dict] = []
    for floor in floors:
        for margin in margins:
            fdr, recall, prec = _rates(keeps, drops, floor, margin)
            sweep.append({"floor": round(float(floor), 4), "margin": round(float(margin), 4),
                          "point": None, "false_drop_rate": fdr,
                          "drop_recall": recall, "drop_precision": prec})

    # tie-break: higher recall, then precision, then smaller floor/margin (least aggressive)
    def rank(r):
        return (r["drop_recall"], r["drop_precision"], -r["floor"], -r["margin"])

    points: dict[str, dict] = {}
    prod = [r for r in sweep if r["false_drop_rate"] <= production_false_drop]
    if prod:
        best = max(prod, key=rank)
        best["point"] = "production" if best["point"] is None else best["point"] + ",production"
        points["production"] = {"floor": best["floor"], "margin": best["margin"]}
    mnr = [r for r in sweep if r["drop_precision"] >= miner_precision]
    if mnr:
        best = max(mnr, key=rank)
        best["point"] = "miner" if best["point"] is None else best["point"] + ",miner"
        points["miner"] = {"floor": best["floor"], "margin": best["margin"]}
    return sweep, points


def negatives_index_is_current(meta: dict, anchor: list[str], model_id: str) -> bool:
    """True iff the stored negatives index matches the current anchor half AND embedding model.

    Guards against reusing a stale index whose surfaces overlap the eval half — that would
    leak eval negatives into the anchors and inflate the sweep's separability — and against
    one embedded by a different model than the calibration will record (its vectors live in a
    different space, so the neg-similarity that drives every margin drop is meaningless).
    """
    return (list(meta.get("surfaces") or []) == list(anchor)
            and meta.get("model_id") == model_id)


def _frange(lo: float, hi: float, step: float) -> list[float]:
    n = round((hi - lo) / step)
    return [round(lo + i * step, 4) for i in range(n + 1)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profiles", default=str(lp.DEFAULT_PROFILE_PATH))
    ap.add_argument("--negatives", default=str(DEFAULT_NEGATIVES_PATH))
    ap.add_argument("--out", default=str(DEFAULT_CALIBRATION_PATH))
    ap.add_argument("--production-false-drop", type=float, default=0.001)
    ap.add_argument("--miner-precision", type=float, default=0.99)
    args = ap.parse_args()

    profiles_path = Path(args.profiles)
    index = load_embindex(str(_index_path_for(profiles_path)), str(profiles_path))
    if index is None:
        raise SystemExit(f"no usable embindex for {profiles_path} (build it first)")
    model_id = index.model_id

    anchor, eval_half = anchor_seed_split(seed_negative_surfaces())
    negatives = _load_negatives(args.negatives)
    if negatives is None or not negatives_index_is_current(negatives[1], anchor, model_id):
        if negatives is not None:
            print(f"negatives index stale (surfaces/model != current anchor half) — "
                  f"rebuilding from {len(anchor)} anchor surfaces")
        build_negative_index(out_path=args.negatives, model_id=model_id, surfaces=anchor)
        negatives = _load_negatives(args.negatives)
    if negatives is None:
        raise SystemExit(f"could not build/load negatives index at {args.negatives}")
    neg_vectors, neg_meta = negatives

    # eval set: keeps carry their own profile type; each is scored against ONLY that type's
    # rows, exactly as gate_spans does at runtime.
    artifact = lp.load_profiles(profiles_path)
    keep_pairs = sorted({(s, rt) for rt in _NOISE_FILTER_TYPES
                         for canonical, row in artifact.get("profiles", {}).get(rt, {}).items()
                         for s in (canonical, *row.get("aliases", []))})
    drop_surfaces = eval_half
    assert set(anchor).isdisjoint(drop_surfaces), "anchor/eval negatives overlap"
    noise_types = sorted(_NOISE_FILTER_TYPES)

    model = _st_model(model_id)
    embed_fn = lambda t: model.encode(t, normalize_embeddings=True)
    keeps = keep_scores_per_type(keep_pairs, index, neg_vectors, embed_fn)
    drops = drop_scores_worst_case(drop_surfaces, index, neg_vectors, embed_fn, noise_types)

    floors = _frange(0.4, 0.8, 0.05)
    margins = _frange(0.0, 0.4, 0.05)
    sweep, points = choose_points(keeps, drops, floors=floors, margins=margins,
                                  production_false_drop=args.production_false_drop,
                                  miner_precision=args.miner_precision)

    # variant-surface slice: cheap near-misses of a seeded sample of keeps, scored per-type
    # like runtime. Reported only (measure-first) — if variant_false_drop_rate is nonzero the
    # documented follow-up is wiring the semantic matcher into span_gate, NOT done here.
    sample = random.Random(0).sample(keep_pairs, min(300, len(keep_pairs)))
    variant_pairs = sorted({(v, rt) for surface, rt in sample for v in _variants(surface)})
    variant_scores = keep_scores_per_type(variant_pairs, index, neg_vectors, embed_fn)
    prod = points.get("production")
    variant_fdr = (_rates(variant_scores, [], prod["floor"], prod["margin"])[0]
                   if prod and variant_scores else None)

    out = {"schema_version": SCHEMA_VERSION, "model_id": model_id,
           "seed_rule": neg_meta.get("seed_rule", "sha256-even-anchor"),
           "eval": {"keeps": len(keeps), "drops": len(drops)},
           "variant_eval": {"n_variants": len(variant_scores),
                            "variant_false_drop_rate": variant_fdr},
           "sweep": sweep, "points": points}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    n_pos = sum(len(index.type_rows(t)) for t in noise_types)
    print(f"eval: keeps={len(keeps)} drops={len(drops)} "
          f"pos_anchors={n_pos} "
          f"neg_anchors={neg_vectors.shape[0] if neg_vectors.size else 0}")
    print(f"variant_eval: n_variants={len(variant_scores)} "
          f"variant_false_drop_rate="
          f"{'n/a (no production point)' if variant_fdr is None else f'{variant_fdr:.4f}'}")
    for name, pt in points.items():
        row = next(r for r in sweep if r["floor"] == pt["floor"] and r["margin"] == pt["margin"])
        print(f"{name}: floor={pt['floor']} margin={pt['margin']} "
              f"false_drop_rate={row['false_drop_rate']:.4f} "
              f"drop_recall={row['drop_recall']:.4f} drop_precision={row['drop_precision']:.4f}")
    for name in ("production", "miner"):
        if name not in points:
            print(f"{name}: DISABLED (bar unreachable) — gate fail-opens")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
