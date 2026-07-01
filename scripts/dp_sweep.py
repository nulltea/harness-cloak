"""ε sweep on the default φ with every probe + attack, for selecting HTML metrics.

Fixed: φ = data/vocab (qwen3-embedding), faithful per-dim Δφ (ε is the only knob),
gen Y = Qwen3.6-35B-A3B (remote), ext X = gemma-4-E4B (local).
Per ε computes: mechanism diagnostics, leakage probes, MI probes, utility probes
(incl. remote-output u_p_control, Δutil gen_marginal, and paper metrics diversity /
coherence / MAUVE vs gold — MAUVE corpus-level, skipped on tiny corpora), attacks.
LLM calls are issued concurrently (the proxy batches them); GPU probes stay sequential.
Run: PYTHONPATH=src python scripts/dp_sweep.py --corpus corpora/cnndm.jsonl --eps 1,3,6,10,14
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from inferdpt import diagnostics
from inferdpt.attacks import embedding_inversion, mask_bert
from inferdpt.embeddings import VocabEmbeddings
from inferdpt.extraction import extract
from inferdpt.llm import LLMClient
from inferdpt.pipeline import gen_prompt, load_corpus
from inferdpt.probes import leakage, mi, utility
from inferdpt.probes._common import content_words

NOTHINK = {"chat_template_kwargs": {"enable_thinking": False}}


def is_content(w: str) -> bool:
    return bool(content_words(w))


def _pmap(fn, items, workers):
    """Map fn over items concurrently, preserving order. For the remote LLM calls only —
    the proxy batches concurrent requests (~7x at 8 workers). GPU probes must NOT use this."""
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, items))


def run(cache, eps_grid, corpus, gen_model, ext_model, out, seed=0, workers=8, limit=0,
        cr_target=0.01):
    docs, golds = load_corpus(corpus)  # golds = gold human continuations (MAUVE ref), or None
    if limit:
        docs = docs[:limit]
        golds = golds[:limit] if golds is not None else None
    ve = VocabEmbeddings.load(cache)
    from inferdpt.rantext import Perturber, calibrate_noise_fn
    # Faithful per-φ Z: refit to a fixed |C_r|/|V| target (paper Appendix B) instead of
    # ada-002's curve-fit constants. Same target across φ ⇒ equal operating point ⇒ fair A/B.
    noise_fn = calibrate_noise_fn(ve, target=cr_target)
    perturber = Perturber(ve, noise_fn=noise_fn)
    print(f"calibrated Z={noise_fn(2.0):.2f} for |C_r|/|V|≈{cr_target} ({cache})")
    rng = np.random.default_rng(seed)

    Y = LLMClient(gen_model, temperature=0.7, max_tokens=256, extra_body=NOTHINK)
    X = LLMClient(ext_model, temperature=0.3, max_tokens=256, extra_body=NOTHINK)
    controls = _pmap(lambda d: Y.generate(gen_prompt(d)), docs, workers)  # ε-independent
    ablates = _pmap(lambda d: extract(d, "", X), docs, workers)           # ε-independent (prefix-only)
    util_ablate = utility.cosine_pairs(list(zip(docs, ablates)))  # batched SimCSE
    swnw_words = sorted({w for d in docs for w in content_words(d) if w in ve.index})[:40]

    geom = {**diagnostics.anisotropy(ve.matrix, rng), **diagnostics.concentration(ve.matrix, rng)}
    rows = []
    result = {"phi": cache, "gen_model": gen_model, "ext_model": ext_model, "docs": len(docs),
              "cr_target": cr_target, "calibrated_Z": round(noise_fn(2.0), 3),
              "geometry": {k: round(v, 4) for k, v in geom.items()}, "sweep": rows}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    for eps in eps_grid:
        mech = diagnostics.mechanism(ve, [eps], rng, sample=200, noise_fn=noise_fn)[0]
        swnw = leakage.s_w_n_w(perturber, swnw_words, eps, runs=100)
        tmi = mi.token_channel_mi(perturber, swnw_words, eps, runs=150)["mi_bits"]
        nmi = mi.ngram_mi(perturber, docs, eps, n=1)["mi_bits"]

        per = {k: [] for k in ["overlap", "pii_leak_recall", "utility", "utility_control",
                               "u_p_control", "gen_marginal", "diversity", "diversity_gen_p",
                               "coherence_gen_p", "utility_rerank", "utility_control_rerank",
                               "pii_recon_recall", "inv@1", "inv@10", "mlm_top1", "mlm_top5"]}
        # Phase 0 — perturb every doc (sequential, fast CPU; avoids dist-cache races).
        kepts = [[(r, p) for r, p in perturber.perturb_aligned(doc, eps, seed=seed) if p is not None]
                 for doc in docs]
        doc_ps = [" ".join(p for _, p in kept) for kept in kepts]

        # Phase A — gen + extract per doc, issued CONCURRENTLY (proxy batches the LLM calls).
        def _gen_extract(i):
            gen_p = Y.generate(gen_prompt(doc_ps[i]))
            return gen_p, extract(docs[i], gen_p, X)
        ge = _pmap(_gen_extract, range(len(docs)), workers)

        # Phase B — probes + attacks (sequential; GPU models are not thread-safe).
        gen_ps = [g for g, _ in ge]
        outs = [o for _, o in ge]
        # Batched SimCSE: ONE embed of all unique texts for the 4 cosine metrics (was 4N tiny calls).
        N = len(docs)
        cos_all = utility.cosine_pairs(list(zip(docs, outs)) + list(zip(controls, outs))
                                       + list(zip(controls, gen_ps)) + list(zip(docs, gen_ps)))
        util_l, uctrl_l, upc_l, coh_l = cos_all[:N], cos_all[N:2*N], cos_all[2*N:3*N], cos_all[3*N:]
        for i, (doc, ctrl, u_ablate) in enumerate(zip(docs, controls, util_ablate)):
            kept, doc_p, gen_p, out_txt = kepts[i], doc_ps[i], gen_ps[i], outs[i]
            per["utility"].append(util_l[i]); per["utility_control"].append(uctrl_l[i])
            per["u_p_control"].append(upc_l[i]); per["coherence_gen_p"].append(coh_l[i])
            per["gen_marginal"].append(util_l[i] - u_ablate)                    # Δutil (ablation ε-independent)
            per["diversity"].append(utility.diversity(out_txt))                 # paper: SimCTG diversity
            per["diversity_gen_p"].append(utility.diversity(gen_p))             # remote-output diversity
            per["overlap"].append(leakage.overlap(doc, doc_p))
            per["pii_leak_recall"].append(leakage.pii_semantic_leakage(doc, doc_p)["recall"])
            per["utility_rerank"].append(utility.utility_rerank(doc, out_txt))         # reranker (remote)
            per["utility_control_rerank"].append(utility.utility_control_rerank(ctrl, out_txt))
            per["pii_recon_recall"].append(utility.pii_reconstruction_recall(doc, out_txt)["recall"])
            pairs = [(r, p) for r, p in kept if is_content(r) and p in ve.index]
            inv = embedding_inversion.invert(pairs, ve, ks=(1, 10))
            per["inv@1"].append(inv["recovery@1"]); per["inv@10"].append(inv["recovery@10"])
            mlm = mask_bert.reconstruct([p for _, p in kept], [r for r, _ in kept], is_content)
            per["mlm_top1"].append(mlm["top1_recovery"]); per["mlm_top5"].append(mlm["top5_recovery"])

        # MAUVE vs gold human continuations (paper-faithful); falls back to control if no gold.
        mauve = utility.mauve_score(outs, golds if golds is not None else controls)
        row = {"eps": eps,
               "Cr_pct": round(100 * mech["|C_r|/V"], 1),
               "repl_cos": round(mech["cos(orig,repl)"], 3),
               "sampling_entropy": round(mech["norm_entropy"], 3),
               "S_w": round(swnw["S_w"], 3), "N_w": round(swnw["N_w"], 1),
               "overlap": round(float(np.mean(per["overlap"])), 3),
               "pii_leak_recall": round(float(np.nanmean(per["pii_leak_recall"])), 3),
               "token_MI_bits": round(tmi, 3), "ngram_MI_bits": round(nmi, 3),
               "utility": round(float(np.mean(per["utility"])), 3),
               "utility_control": round(float(np.mean(per["utility_control"])), 3),
               "u_p_control": round(float(np.mean(per["u_p_control"])), 3),
               "gen_marginal": round(float(np.mean(per["gen_marginal"])), 3),
               "diversity": round(float(np.mean(per["diversity"])), 3),
               "diversity_gen_p": round(float(np.mean(per["diversity_gen_p"])), 3),
               "coherence_gen_p": round(float(np.mean(per["coherence_gen_p"])), 3),
               "mauve": mauve["mauve"],
               "utility_rerank": round(float(np.mean(per["utility_rerank"])), 3),
               "utility_control_rerank": round(float(np.mean(per["utility_control_rerank"])), 3),
               "pii_recon_recall": round(float(np.nanmean(per["pii_recon_recall"])), 3),
               "inv@1": round(float(np.mean(per["inv@1"])), 3),
               "inv@10": round(float(np.mean(per["inv@10"])), 3),
               "mlm_top1": round(float(np.mean(per["mlm_top1"])), 3),
               "mlm_top5": round(float(np.mean(per["mlm_top5"])), 3)}
        rows.append(row)
        Path(out).write_text(json.dumps(result, indent=2))  # incremental: persist after each ε
        print(f"eps={eps:>3} | Cr%={row['Cr_pct']:>5} repl_cos={row['repl_cos']} "
              f"tokenMI={row['token_MI_bits']} S_w={row['S_w']} overlap={row['overlap']} "
              f"piiLeak={row['pii_leak_recall']} | util={row['utility']} uctrl={row['utility_control']} "
              f"u_pctrl={row['u_p_control']} genMrg={row['gen_marginal']} div={row['diversity']} "
              f"cohGenP={row['coherence_gen_p']} mauve={row['mauve']} "
              f"piiRec={row['pii_recon_recall']} | inv@10={row['inv@10']} mlm_top1={row['mlm_top1']} [saved]")

    print(f"\nsaved {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/vocab")
    ap.add_argument("--eps", default="2,6,10")
    ap.add_argument("--corpus", default="corpora/cnndm.jsonl")
    ap.add_argument("--limit", type=int, default=60, help="use first N corpus docs (0 = all)")
    ap.add_argument("--gen-model", default="Qwen3.6-35B-A3B")
    ap.add_argument("--ext-model", default="gemma 4 (E4B)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cr-target", type=float, default=0.01, help="|C_r|/|V| target for per-φ Z calibration")
    ap.add_argument("--out", default="results/dp_sweep.json")
    args = ap.parse_args()
    run(args.cache, [float(x) for x in args.eps.split(",")],
        args.corpus, args.gen_model, args.ext_model, args.out, workers=args.workers,
        limit=args.limit, cr_target=args.cr_target)
