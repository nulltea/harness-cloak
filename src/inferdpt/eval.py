"""End-to-end A/B evaluation harness for RANTEXT embedding backends.

Per φ backend (faithful per-dim Δφ, no per-φ calibration — |C_r| is a measured outcome):
  geometry  — anisotropy, distance concentration, mechanism retention cos(orig,repl)
  leakage   — S_w/N_w (intrinsic), overlap, PII semantic leakage (↓ better)
  utility   — utility, utility_control, u_p_control (remote-output), gen_marginal, PII recon (↑ better)
  attacks   — (--attacks) embedding inversion recovery@K, Mask/BERT recovery+graded

All φ caches must be ROW-ALIGNED (same vocab order) for a fair A/B; the sub-vocab set below
is (7,913 whole-word tokens). Runs every φ at the SAME ε with its own per-dim Δφ — |C_r| is a
measured outcome, never matched by a per-φ knob.

Run: PYTHONPATH=src python src/inferdpt/eval.py --corpus corpora/cnndm.jsonl --eps 6 --attacks
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from inferdpt import diagnostics
from inferdpt.attacks import embedding_inversion, mask_bert
from inferdpt.embeddings import VocabEmbeddings
from inferdpt.extraction import extract
from inferdpt.pipeline import default, gen_prompt, load_corpus, pmap
from inferdpt.probes import leakage, mi, utility
from inferdpt.probes._common import content_words
from inferdpt.rantext import Perturber


def _is_content(w: str) -> bool:
    return bool(content_words(w))


def eval_backend(cache: str, eps: float, docs: list[str], controls: list[str],
                 util_ablate: list[float], gen, ext, *, attacks: bool, seed: int = 0,
                 golds: list[str] | None = None, workers: int = 8) -> dict:
    ve = VocabEmbeddings.load(cache)
    rng = np.random.default_rng(seed)
    # Paper-faithful, identical ε-LDP for every φ (per-dim Δφ, no per-φ tuning);
    # |C_r| varies as a measured OUTCOME.
    perturber = Perturber(ve)

    geo = diagnostics.anisotropy(ve.matrix, rng)
    conc = diagnostics.concentration(ve.matrix, rng)
    mech = diagnostics.mechanism(ve, [eps], rng, sample=200)[0]

    # intrinsic S_w/N_w over a sample of corpus content words present in V
    corpus_words = sorted({w for d in docs for w in content_words(d) if w in ve.index})[:40]
    swnw = leakage.s_w_n_w(perturber, corpus_words, eps, runs=100)
    mi_bits = mi.token_channel_mi(perturber, corpus_words, eps, runs=100)["mi_bits"]

    # Phase 0 — perturb (sequential CPU). Phase A — gen+extract concurrently (proxy batches).
    kepts = [[(raw, rep) for raw, rep in perturber.perturb_aligned(doc, eps, seed=seed) if rep is not None]
             for doc in docs]
    doc_ps = [" ".join(rep for _, rep in kept) for kept in kepts]

    def _gen_extract(i):
        gen_p = gen.generate(gen_prompt(doc_ps[i]))
        return gen_p, extract(docs[i], gen_p, ext)
    ge = pmap(_gen_extract, range(len(docs)), workers)

    # Phase B — probes + attacks (sequential; GPU models not thread-safe).
    gen_ps = [g for g, _ in ge]
    outs = [o for _, o in ge]
    # Batched SimCSE: ONE embed of all unique texts for the 4 cosine metrics (was 4N tiny calls).
    N = len(docs)
    cos_all = utility.cosine_pairs(list(zip(docs, outs)) + list(zip(controls, outs))
                                   + list(zip(controls, gen_ps)) + list(zip(docs, gen_ps)))
    util_l, uctrl_l, upc_l, coh_l = cos_all[:N], cos_all[N:2*N], cos_all[2*N:3*N], cos_all[3*N:]
    rows: list[dict] = []
    for i, (doc, control, u_ab) in enumerate(zip(docs, controls, util_ablate)):
        kept, doc_p, gen_p, out = kepts[i], doc_ps[i], gen_ps[i], outs[i]
        r = {
            "overlap": leakage.overlap(doc, doc_p),
            "pii_leak": leakage.pii_semantic_leakage(doc, doc_p)["degree"],
            "utility": util_l[i],
            "utility_control": uctrl_l[i],
            "u_p_control": upc_l[i],                                     # REMOTE-output utility (no extractor)
            "gen_marginal": util_l[i] - u_ab,                           # Δutil: marginal value of Gen_p (ablation φ-indep)
            "diversity": utility.diversity(out),                        # paper: SimCTG diversity
            "coherence_gen_p": coh_l[i],                               # paper: coherence, on Gen_p
            "pii_recon": utility.pii_reconstruction_recall(doc, out)["degree"],
        }
        if attacks:
            pairs = [(raw, rep) for raw, rep in kept if _is_content(raw) and rep in ve.index]
            r["inv@10"] = embedding_inversion.invert(pairs, ve, ks=(10,))["recovery@10"]
            mlm = mask_bert.reconstruct([rep for _, rep in kept], [raw for raw, _ in kept], _is_content)
            r["mlm_top1"] = mlm["top1_recovery"]
        rows.append(r)

    agg = {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0]}
    mauve = utility.mauve_score(outs, golds if golds is not None else controls)
    return {"cache": cache,
            "Cr/V": round(mech["|C_r|/V"], 3),
            "anisotropy": round(geo["mean_cos_random_pairs"], 3),
            "rel_spread": round(conc["rel_spread_std/mean"], 4),
            "retention": round(mech["cos(orig,repl)"], 3),
            "S_w": round(swnw["S_w"], 3), "N_w": round(swnw["N_w"], 1),
            "mi_bits": round(mi_bits, 3), "mauve": mauve["mauve"],
            **{k: round(v, 3) for k, v in agg.items()}}


def main(caches, eps, corpus, out, attacks, workers=8, limit=0) -> None:
    docs, golds = load_corpus(corpus)
    if limit:
        docs = docs[:limit]
        golds = golds[:limit] if golds is not None else None
    pipe = default()
    gen, ext = pipe.gen_client, pipe.ext_client
    # φ-independent, computed once and shared across all φ: the non-private control
    # generation, and the Gen_p-ablation extraction (extractor on the true prefix only).
    controls = pmap(lambda d: gen.generate(gen_prompt(d)), docs, workers)
    ablates = pmap(lambda d: extract(d, "", ext), docs, workers)
    util_ablate = utility.cosine_pairs(list(zip(docs, ablates)))  # batched SimCSE
    rows = [eval_backend(c, eps, docs, controls, util_ablate, gen, ext, attacks=attacks,
                         golds=golds, workers=workers) for c in caches]

    cols = list(rows[0].keys())
    print(f"\nε={eps}  docs={len(docs)}  attacks={attacks}\n")
    print("  ".join(f"{c:>22}" if c == "cache" else f"{c:>12}" for c in cols))
    for r in rows:
        print("  ".join(f"{str(r[c]):>22}" if c == "cache" else f"{str(r[c]):>12}" for c in cols))
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(rows, indent=2))
        print(f"\nsaved {out}")


if __name__ == "__main__":
    import argparse

    # Row-aligned 7,913-token sub-vocab φ set (Lever-2 e2e backends from embedding-map.md).
    # 6-φ set: drops the within-family qwen3-0.6B/4B matrices (dimension controls already
    # settled — anisotropy, not dim, drives inv@10); pythia (isotropic) + gemma (anisotropic)
    # span the matrix range.
    DEFAULT_CACHES = ",".join("data/" + c for c in [
        "vocab_qwen_sub",          # qwen3-embedding (baseline)
        "vocab_glove", "vocab_cf", "vocab_phrasebert",
        "vocab_pythia410m", "vocab_gemma4e4b_matrix",
    ])
    ap = argparse.ArgumentParser()
    ap.add_argument("--caches", default=DEFAULT_CACHES,
                    help="comma-separated row-aligned φ cache stems")
    ap.add_argument("--eps", type=float, default=6.0, help="fixed ε for all φ (operating point)")
    ap.add_argument("--corpus", default="corpora/cnndm.jsonl")
    ap.add_argument("--limit", type=int, default=60, help="use first N corpus docs (0 = all)")
    ap.add_argument("--out", default="results/phi_sweep.json")
    ap.add_argument("--attacks", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    main(args.caches.split(","), args.eps, args.corpus, args.out, args.attacks,
         workers=args.workers, limit=args.limit)
