"""ε sweep on the default φ with every probe + attack, for selecting HTML metrics.

Fixed: φ = data/vocab (qwen3-embedding), noise_scale rescaled (ε acts), Y = X = gemma-4-E4B.
Per ε computes: mechanism diagnostics, leakage probes, MI probes, utility probes, attacks.
Run: PYTHONPATH=src python scripts/dp_sweep.py --eps 1,3,6,10,14 --out results/dp_sweep.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from inferdpt import diagnostics
from inferdpt.attacks import embedding_inversion, mask_bert
from inferdpt.embeddings import VocabEmbeddings
from inferdpt.extraction import extract
from inferdpt.llm import LLMClient
from inferdpt.pipeline import GENERATION_INSTRUCTION
from inferdpt.probes import leakage, mi, utility
from inferdpt.probes._common import content_words

NOTHINK = {"chat_template_kwargs": {"enable_thinking": False}}


def is_content(w: str) -> bool:
    return bool(content_words(w))


def run(cache, eps_grid, noise_scale, corpus, model, out, seed=0):
    docs = [ln.strip() for ln in Path(corpus).read_text().splitlines() if ln.strip()]
    ve = VocabEmbeddings.load(cache)
    from inferdpt.rantext import Perturber
    perturber = Perturber(ve, noise_scale=noise_scale)
    rng = np.random.default_rng(seed)

    Y = LLMClient(model, temperature=0.7, max_tokens=256, extra_body=NOTHINK)
    X = LLMClient(model, temperature=0.3, max_tokens=256, extra_body=NOTHINK)
    controls = [Y.generate(f"{GENERATION_INSTRUCTION}\n\n- Prefix Text: {d}") for d in docs]  # ε-independent
    swnw_words = sorted({w for d in docs for w in content_words(d) if w in ve.index})[:40]

    geom = {**diagnostics.anisotropy(ve.matrix, rng), **diagnostics.concentration(ve.matrix, rng)}
    rows = []
    for eps in eps_grid:
        mech = diagnostics.mechanism(ve, [eps], noise_scale, rng, sample=200)[0]
        swnw = leakage.s_w_n_w(perturber, swnw_words, eps, runs=100)
        tmi = mi.token_channel_mi(perturber, swnw_words, eps, runs=150)["mi_bits"]
        nmi = mi.ngram_mi(perturber, docs, eps, n=1)["mi_bits"]

        per = {k: [] for k in ["overlap", "pii_leak_recall", "utility", "utility_control",
                               "utility_rerank", "utility_control_rerank",
                               "pii_recon_recall", "inv@1", "inv@10", "mlm_top1", "mlm_top5"]}
        for doc, ctrl in zip(docs, controls):
            aligned = perturber.perturb_aligned(doc, eps, seed=seed)
            kept = [(r, p) for r, p in aligned if p is not None]
            doc_p = " ".join(p for _, p in kept)
            out_txt = extract(doc, Y.generate(f"{GENERATION_INSTRUCTION}\n\n- Prefix Text: {doc_p}"), X)
            per["overlap"].append(leakage.overlap(doc, doc_p))
            per["pii_leak_recall"].append(leakage.pii_semantic_leakage(doc, doc_p)["recall"])
            per["utility"].append(utility.utility(doc, out_txt))                      # SimCSE cos
            per["utility_control"].append(utility.utility_control(ctrl, out_txt))      # SimCSE cos
            per["utility_rerank"].append(utility.utility_rerank(doc, out_txt))         # reranker
            per["utility_control_rerank"].append(utility.utility_control_rerank(ctrl, out_txt))
            per["pii_recon_recall"].append(utility.pii_reconstruction_recall(doc, out_txt)["recall"])
            pairs = [(r, p) for r, p in kept if is_content(r) and p in ve.index]
            inv = embedding_inversion.invert(pairs, ve, ks=(1, 10))
            per["inv@1"].append(inv["recovery@1"]); per["inv@10"].append(inv["recovery@10"])
            mlm = mask_bert.reconstruct([p for _, p in kept], [r for r, _ in kept], is_content)
            per["mlm_top1"].append(mlm["top1_recovery"]); per["mlm_top5"].append(mlm["top5_recovery"])

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
               "utility_rerank": round(float(np.mean(per["utility_rerank"])), 3),
               "utility_control_rerank": round(float(np.mean(per["utility_control_rerank"])), 3),
               "pii_recon_recall": round(float(np.nanmean(per["pii_recon_recall"])), 3),
               "inv@1": round(float(np.mean(per["inv@1"])), 3),
               "inv@10": round(float(np.mean(per["inv@10"])), 3),
               "mlm_top1": round(float(np.mean(per["mlm_top1"])), 3),
               "mlm_top5": round(float(np.mean(per["mlm_top5"])), 3)}
        rows.append(row)
        print(f"eps={eps:>3} | Cr%={row['Cr_pct']:>5} repl_cos={row['repl_cos']} "
              f"tokenMI={row['token_MI_bits']} S_w={row['S_w']} overlap={row['overlap']} "
              f"piiLeak={row['pii_leak_recall']} | util={row['utility']} uctrl={row['utility_control']} "
              f"piiRec={row['pii_recon_recall']} | inv@10={row['inv@10']} mlm_top1={row['mlm_top1']}")

    result = {"phi": cache, "noise_scale": noise_scale, "model": model, "docs": len(docs),
              "geometry": {k: round(v, 4) for k, v in geom.items()}, "sweep": rows}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(result, indent=2))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/vocab")
    ap.add_argument("--eps", default="1,3,6,10,14")
    ap.add_argument("--noise-scale", type=float, default=0.38)
    ap.add_argument("--corpus", default="corpora/dev.txt")
    ap.add_argument("--model", default="gemma 4 (E4B)")
    ap.add_argument("--out", default="results/dp_sweep.json")
    args = ap.parse_args()
    run(args.cache, [float(x) for x in args.eps.split(",")], args.noise_scale,
        args.corpus, args.model, args.out)
