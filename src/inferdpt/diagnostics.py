"""Quantify the geometry limitations of a RANTEXT vocab-embedding cache.

Measures, on the actual mechanism (via Perturber.candidates):
  1. Embedding spread / anisotropy  — random-pair cosine, top-PC variance share
  2. Curse of dimensionality        — distance concentration; |C_r|/|V| & sampling
                                       entropy & radius vs ε (mechanism collapse)
  3. Density variation              — per-token k-NN radius spread (CoV)
Plus a utility proxy: cosine(original, replacement) vs ε.

Run: PYTHONPATH=src python src/inferdpt/diagnostics.py --cache data/vocab --noise-scale 1.0
"""

from __future__ import annotations

import numpy as np

from inferdpt.embeddings import VocabEmbeddings
from inferdpt.rantext import Perturber


def anisotropy(E: np.ndarray, rng, n_pairs: int = 20000) -> dict:
    n = len(E)
    i = rng.integers(0, n, n_pairs)
    j = rng.integers(0, n, n_pairs)
    cos = np.einsum("ij,ij->i", E[i], E[j])  # unit-norm → dot == cosine
    centered = E - E.mean(0)
    cov = (centered.T @ centered) / len(E)
    ev = np.sort(np.linalg.eigvalsh(cov))[::-1]
    ev = ev / ev.sum()
    return {
        "mean_cos_random_pairs": float(cos.mean()),
        "mean_abs_cos": float(np.abs(cos).mean()),
        "top1_pc_var": float(ev[0]),
        "top10_pc_var": float(ev[:10].sum()),
    }


def concentration(E: np.ndarray, rng, n_pairs: int = 20000) -> dict:
    i = rng.integers(0, len(E), n_pairs)
    j = rng.integers(0, len(E), n_pairs)
    d = np.linalg.norm(E[i] - E[j], axis=1)
    return {
        "mean": float(d.mean()),
        "std": float(d.std()),
        "rel_spread_std/mean": float(d.std() / d.mean()),
        "p1": float(np.percentile(d, 1)),
        "p50": float(np.percentile(d, 50)),
        "p99": float(np.percentile(d, 99)),
    }


def density(E: np.ndarray, rng, sample: int = 300, ks=(10, 100)) -> dict:
    idx = rng.choice(len(E), sample, replace=False)
    out = {}
    knn = {k: [] for k in ks}
    for t in idx:
        d = np.sort(np.linalg.norm(E - E[t], axis=1))
        for k in ks:
            knn[k].append(d[k])  # d[0]==0 (self)
    for k in ks:
        a = np.array(knn[k])
        out[f"knn{k}_mean"] = float(a.mean())
        out[f"knn{k}_CoV"] = float(a.std() / a.mean())  # heterogeneity of local density
    return out


def mechanism(ve: VocabEmbeddings, epsilons, noise_scale: float, rng,
              sample: int = 200, trials: int = 3) -> list[dict]:
    p = Perturber(ve, _NoTok(), noise_scale=noise_scale)
    V = len(ve.vocab)
    idx = rng.choice(V, sample, replace=False)
    rows = []
    for eps in epsilons:
        crfrac, ent, rad, util, empty = [], [], [], [], 0
        for t in idx:
            for _ in range(trials):
                cand, probs, radius = p.candidates(int(t), eps, rng)
                rad.append(radius)
                if cand is None:
                    empty += 1
                    continue
                crfrac.append(len(cand) / V)
                h = -(probs * np.log(probs)).sum()
                ent.append(h / np.log(len(cand)) if len(cand) > 1 else 0.0)
                repl = int(rng.choice(cand, p=probs))
                util.append(float(ve.matrix[t] @ ve.matrix[repl]))  # cosine orig vs replacement
        rows.append({
            "eps": eps,
            "|C_r|/V": float(np.mean(crfrac)) if crfrac else 0.0,
            "norm_entropy": float(np.mean(ent)) if ent else 0.0,
            "radius": float(np.mean(rad)),
            "cos(orig,repl)": float(np.mean(util)) if util else float("nan"),
            "empty_%": 100.0 * empty / (sample * trials),
        })
    return rows


class _NoTok:
    def surfaces(self, t):  # diagnostics never tokenise
        return t.split()


def run(cache: str, epsilons, noise_scale: float, seed: int = 0) -> None:
    ve = VocabEmbeddings.load(cache)
    E = ve.matrix
    rng = np.random.default_rng(seed)
    print(f"cache={cache}  |V|={len(ve.vocab)}  dim={E.shape[1]}  Δφ={ve.sensitivity:.4f}  noise_scale={noise_scale}\n")

    print("[1] Embedding spread / anisotropy  (isotropic ⇒ mean_cos≈0, top1_pc_var small)")
    for k, v in anisotropy(E, rng).items():
        print(f"    {k:24s} {v:.4f}")

    print("\n[2a] Distance concentration  (curse of dim ⇒ rel_spread→0, p1≈p99)")
    for k, v in concentration(E, rng).items():
        print(f"    {k:24s} {v:.4f}")

    print("\n[3] Density variation  (uneven density ⇒ high CoV)")
    for k, v in density(E, rng).items():
        print(f"    {k:24s} {v:.4f}")

    print("\n[2b] Mechanism vs ε  (collapse ⇒ |C_r|/V→1, norm_entropy→1, radius flat, cos(orig,repl)→random)")
    print(f"    {'eps':>5} {'|C_r|/V':>9} {'norm_ent':>9} {'radius':>8} {'cos(o,r)':>9} {'empty%':>7}")
    for r in mechanism(ve, epsilons, noise_scale, rng):
        print(f"    {r['eps']:>5.1f} {r['|C_r|/V']:>9.3f} {r['norm_entropy']:>9.3f} "
              f"{r['radius']:>8.3f} {r['cos(orig,repl)']:>9.3f} {r['empty_%']:>7.1f}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/vocab")
    ap.add_argument("--noise-scale", type=float, default=1.0, help="1.0 = faithful reference")
    ap.add_argument("--epsilons", default="1,3,6,10,14")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(args.cache, [float(x) for x in args.epsilons.split(",")], args.noise_scale, args.seed)
