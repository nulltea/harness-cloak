"""End-to-end A/B evaluation harness for RANTEXT embedding backends.

Per φ backend: auto-calibrate noise to a target |C_r|, then report
  geometry  — anisotropy, distance concentration, mechanism retention cos(orig,repl)
  leakage   — S_w/N_w (intrinsic), overlap, PII semantic leakage (↓ better)
  utility   — utility, utility_control, PII reconstruction recall (↑ better)
  attacks   — (--attacks) embedding inversion recovery@K, Mask/BERT recovery+graded

Run: PYTHONPATH=src python src/inferdpt/eval.py --caches data/vocab,data/vocab_qwen3matrix [--attacks]
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from inferdpt import diagnostics
from inferdpt.attacks import embedding_inversion, mask_bert
from inferdpt.embeddings import VocabEmbeddings
from inferdpt.extraction import extract
from inferdpt.pipeline import GENERATION_INSTRUCTION, default
from inferdpt.probes import leakage, utility
from inferdpt.probes._common import content_words
from inferdpt.rantext import Perturber


def calibrate(ve: VocabEmbeddings, eps: float, target_frac: float, rng) -> float:
    lo, hi = 0.01, 2.0
    for _ in range(12):
        mid = (lo + hi) / 2
        frac = diagnostics.mechanism(ve, [eps], mid, rng, sample=80, trials=1)[0]["|C_r|/V"]
        lo, hi = (lo, mid) if frac > target_frac else (mid, hi)
    return (lo + hi) / 2


def _is_content(w: str) -> bool:
    return bool(content_words(w))


def eval_backend(cache: str, eps: float, target_frac: float, docs: list[str],
                 controls: list[str], gen, ext, *, attacks: bool, seed: int = 0) -> dict:
    ve = VocabEmbeddings.load(cache)
    rng = np.random.default_rng(seed)
    ns = calibrate(ve, eps, target_frac, rng)
    perturber = Perturber(ve, noise_scale=ns)

    geo = diagnostics.anisotropy(ve.matrix, rng)
    conc = diagnostics.concentration(ve.matrix, rng)
    mech = diagnostics.mechanism(ve, [eps], ns, rng, sample=200)[0]

    # intrinsic S_w/N_w over a sample of corpus content words present in V
    corpus_words = sorted({w for d in docs for w in content_words(d) if w in ve.index})[:40]
    swnw = leakage.s_w_n_w(perturber, corpus_words, eps, runs=100)

    rows: list[dict] = []
    for doc, control in zip(docs, controls):
        aligned = perturber.perturb_aligned(doc, eps, seed=seed)
        kept = [(raw, rep) for raw, rep in aligned if rep is not None]
        doc_p = " ".join(rep for _, rep in kept)
        out = extract(doc, gen.generate(f"{GENERATION_INSTRUCTION}\n\n- Prefix Text: {doc_p}"), ext)
        r = {
            "overlap": leakage.overlap(doc, doc_p),
            "pii_leak": leakage.pii_semantic_leakage(doc, doc_p)["degree"],
            "utility": utility.utility(doc, out),
            "utility_control": utility.utility_control(control, out),
            "pii_recon": utility.pii_reconstruction_recall(doc, out)["degree"],
        }
        if attacks:
            pairs = [(raw, rep) for raw, rep in kept if _is_content(raw) and rep in ve.index]
            r["inv@10"] = embedding_inversion.invert(pairs, ve, ks=(10,))["recovery@10"]
            mlm = mask_bert.reconstruct([rep for _, rep in kept], [raw for raw, _ in kept], _is_content)
            r["mlm_top1"] = mlm["top1_recovery"]
        rows.append(r)

    agg = {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0]}
    return {"cache": cache, "noise_scale": round(ns, 4),
            "anisotropy": round(geo["mean_cos_random_pairs"], 3),
            "rel_spread": round(conc["rel_spread_std/mean"], 4),
            "retention": round(mech["cos(orig,repl)"], 3),
            "S_w": round(swnw["S_w"], 3), "N_w": round(swnw["N_w"], 1),
            **{k: round(v, 3) for k, v in agg.items()}}


def main(caches, eps, target_frac, corpus, out, attacks) -> None:
    docs = [ln.strip() for ln in Path(corpus).read_text().splitlines() if ln.strip()]
    pipe = default()
    gen, ext = pipe.gen_client, pipe.ext_client
    controls = [gen.generate(f"{GENERATION_INSTRUCTION}\n\n- Prefix Text: {d}") for d in docs]
    rows = [eval_backend(c, eps, target_frac, docs, controls, gen, ext, attacks=attacks) for c in caches]

    cols = list(rows[0].keys())
    print(f"\nε={eps}  target|C_r|/V={target_frac}  docs={len(docs)}  attacks={attacks}\n")
    print("  ".join(f"{c:>22}" if c == "cache" else f"{c:>12}" for c in cols))
    for r in rows:
        print("  ".join(f"{str(r[c]):>22}" if c == "cache" else f"{str(r[c]):>12}" for c in cols))
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(rows, indent=2))
        print(f"\nsaved {out}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--caches", default="data/vocab,data/vocab_qwen3matrix")
    ap.add_argument("--eps", type=float, default=3.0)
    ap.add_argument("--target-frac", type=float, default=0.05)
    ap.add_argument("--corpus", default="corpora/dev.txt")
    ap.add_argument("--out", default="results/e2e_ab.json")
    ap.add_argument("--attacks", action="store_true")
    args = ap.parse_args()
    main(args.caches.split(","), args.eps, args.target_frac, args.corpus, args.out, args.attacks)
