"""Mutual-information leakage probes for RANTEXT (see docs/research/mi-probes.md).

token_channel_mi — I(X_i;Y_i) of the per-token channel, computed directly from the known
                   exponential-mechanism conditional (soft-averaged over noise). Intrinsic,
                   no LLM; misses cross-token context.
ngram_mi         — empirical plug-in MI (Miller-Madow corrected) between aligned raw and
                   perturbed n-grams over a corpus; a cheap local-context proxy.

Context-aware MI (V-information via the LLM attacker) is documented but deferred.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np


def _entropy_bits(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def token_channel_mi(perturber, words: list[str], epsilon: float, *,
                     runs: int = 200, prior: dict | None = None, seed: int = 0) -> dict:
    """I(X;Y) in bits for the per-token channel over input tokens `words` (kept if in V).

    p(y|x) is the analytic conditional soft-averaged over `runs` noise draws. `prior`
    weights inputs (default uniform). Returns MI, H(Y), H(Y|X) and per-token leakage.
    """
    ve = perturber.ve
    xs = [w for w in words if w in ve.index]
    if not xs:
        return {"mi_bits": float("nan"), "n_x": 0}
    rng = np.random.default_rng(seed)
    V = len(ve.vocab)
    px = (np.array([prior[w] for w in xs], float) if prior else np.ones(len(xs)))
    px /= px.sum()

    cond = np.zeros((len(xs), V))  # p(y|x), marginalized over noise
    for i, w in enumerate(xs):
        for _ in range(runs):
            cand, probs, _ = perturber.candidates(ve.index[w], epsilon, rng)
            if cand is None:
                cond[i, ve.index[w]] += 1.0  # radius caught nothing → token kept unchanged
            else:
                cond[i, cand] += probs
        cond[i] /= runs

    py = (px[:, None] * cond).sum(0)
    h_y = _entropy_bits(py)
    h_y_given_x = float(sum(px[i] * _entropy_bits(cond[i]) for i in range(len(xs))))
    leaks = []
    for i in range(len(xs)):
        p, m = cond[i], cond[i] > 0
        leaks.append(float((p[m] * np.log2(p[m] / py[m])).sum()))  # KL(p(y|x)||p(y))
    return {
        "mi_bits": h_y - h_y_given_x, "h_y": h_y, "h_y_given_x": h_y_given_x,
        "mean_token_leakage_bits": float(np.average(leaks, weights=px)),
        "max_token_leakage_bits": float(np.max(leaks)),
        "n_x": len(xs), "runs": runs,
    }


def _plugin_mi(xs: list, ys: list) -> dict:
    """Plug-in MI in bits with Miller-Madow bias correction."""
    N = len(xs)
    if N == 0:
        return {"mi_bits": float("nan"), "n_pairs": 0}

    def H(counter: Counter) -> float:
        h = -sum((c / N) * math.log2(c / N) for c in counter.values())
        return h + (len(counter) - 1) / (2 * N * math.log(2))  # Miller-Madow (bits)

    mi = H(Counter(xs)) + H(Counter(ys)) - H(Counter(zip(xs, ys)))
    return {"mi_bits": float(mi), "n_pairs": N,
            "support_x": len(set(xs)), "support_y": len(set(ys))}


def ngram_mi(perturber, docs: list[str], epsilon: float, *,
             n: int = 1, seed: int = 0) -> dict:
    """Empirical MI (bits) between aligned raw and perturbed n-grams over `docs`.
    n=1 ≈ empirical token MI; n≥2 is a local-context proxy (needs many tokens)."""
    xs, ys = [], []
    for d in docs:
        kept = [(r, p) for r, p in perturber.perturb_aligned(d, epsilon, seed=seed) if p is not None]
        raw, per = [r for r, _ in kept], [p for _, p in kept]
        for i in range(len(kept) - n + 1):
            xs.append(tuple(raw[i:i + n]))
            ys.append(tuple(per[i:i + n]))
    return {**_plugin_mi(xs, ys), "n": n}


if __name__ == "__main__":
    # token_channel_mi on a tiny synthetic vocab; ngram via _plugin_mi extremes.
    from inferdpt.embeddings import VocabEmbeddings
    from inferdpt.rantext import Perturber

    rng = np.random.default_rng(0)
    M = rng.standard_normal((24, 16)).astype(np.float32)
    M /= np.linalg.norm(M, axis=1, keepdims=True)
    ve = VocabEmbeddings([f"w{i}" for i in range(24)], M)

    class _NoTok:
        def surfaces(self, t):
            return t.split()

    p = Perturber(ve, _NoTok())
    words = ve.vocab[:12]
    lo = token_channel_mi(p, words, 0.5, runs=80)["mi_bits"]
    hi = token_channel_mi(p, words, 12.0, runs=80)["mi_bits"]
    assert 0 <= lo <= math.log2(24) and 0 <= hi <= math.log2(24), (lo, hi)
    assert hi >= lo - 1e-6, f"more ε should not leak less: {lo=} {hi=}"

    # n-gram extremes: identity ⇒ MI≈H(X); constant target ⇒ MI≈0
    xs = ["a", "b", "c", "d"] * 25
    assert abs(_plugin_mi(xs, xs)["mi_bits"] - 2.0) < 0.1          # H(uniform 4)=2 bits
    assert _plugin_mi(xs, ["z"] * len(xs))["mi_bits"] < 0.1
    print(f"OK  token MI ε0.5={lo:.3f} ε12={hi:.3f} bits")
