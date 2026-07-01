"""RANTEXT perturbation mechanism (InferDPT, Algorithm 2).

Replaces every in-vocabulary token of a document with a semantically-near token
sampled under ε-LDP, via a per-token *random adjacency list*. See
docs/research/rantext.md for the full derivation.
"""

from __future__ import annotations

import math
import random

import numpy as np

from inferdpt.embeddings import VocabEmbeddings
from inferdpt.tokeniser import Tokeniser


def noise_factor(epsilon: float) -> float:
    """Z(ε): calibrates the Laplace scale so the random-radius mechanism is ε-LDP."""
    if epsilon < 2:
        return epsilon
    a, b, c, d = 0.0165, 19.0648, -38.1294, 9.3111
    return a * math.log(b * epsilon + c) + d


_CAL_REFS = ["time", "people", "year", "day", "world", "life", "work", "city",
             "government", "water", "house", "family", "money", "school", "company"]


def calibrate_noise_fn(ve: VocabEmbeddings, *, target: float = 0.015,
                       ref_tokens: list[str] | None = None, samples: int = 256,
                       seed: int = 0):
    """Per-embedding Z calibration — the faithful InferDPT Appendix-B method.

    The paper fits Z(ε) with `scipy.curve_fit` so a reference token's |C_r|/|V| hits a
    target; the published a,b,c,d (`noise_factor`) encode *ada-002's* geometry and are
    meaningless on any other embedding. This refits Z to `ve` so the expected |C_r|/|V|
    over `ref_tokens` equals `target`.

    Returns a constant `noise_fn(eps)->Z`: the paper's Z(ε) is near-flat for ε≥2 (radius
    barely moves), so a single Z reproduces it; ε then drives only the exponential-mechanism
    sharpness. `radius = ‖Laplace(0,Δφ)‖ / Z`, so Z is an exact deterministic root-find on
    fixed base draws (no stochastic curve-fit)."""
    refs = ref_tokens or [w for w in _CAL_REFS if w in ve.index][:10] or ve.vocab[:10]
    p = Perturber(ve, _DummyTok())
    sorted_d = [np.sort(p._dists(ve.index[w])) for w in refs]
    base = np.array([np.linalg.norm(np.random.default_rng(seed + k).laplace(0.0, ve.sensitivity))
                     for k in range(samples)])  # ‖Laplace(0,Δφ)‖ draws; radius = base/Z
    V = len(ve.vocab)

    def frac(Z: float) -> float:  # expected |C_r|/|V| over refs at this Z (deterministic)
        rs = base / Z
        return float(np.mean([np.searchsorted(sd, rs).mean() for sd in sorted_d]) / V)

    lo, hi = 0.5, 1000.0  # frac is monotone-decreasing in Z; bisect to target
    for _ in range(50):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if frac(mid) > target else (lo, mid)
    Z = (lo + hi) / 2
    return lambda eps: Z


class _DummyTok:
    def surfaces(self, t):  # calibration never tokenises
        return t.split()


class Perturber:
    def __init__(
        self,
        vocab: VocabEmbeddings,
        tokeniser: Tokeniser | None = None,
        noise_fn=None,
    ) -> None:
        self.ve = vocab
        self.tok = tokeniser or Tokeniser()
        # Z(ε): ada-002's curve-fit constants (noise_factor) by default. For faithful
        # reproduction on a *different* embedding, pass a per-φ calibrated Z from
        # `calibrate_noise_fn` — the paper's Appendix-B constants only fit ada-002.
        self._noise_fn = noise_fn or noise_factor
        self._delta = vocab.sensitivity  # Δφ: per-dimension range vector [dim]
        # Distance from a token to all of V is independent of noise and ε, so cache it.
        # dist² = ‖a‖² + ‖b‖² − 2a·b → one BLAS matvec (multicore) per token, then memoised.
        self._sqnorm = np.einsum("ij,ij->i", vocab.matrix, vocab.matrix)
        self._dist_cache: dict[int, np.ndarray] = {}

    def _dists(self, idx: int) -> np.ndarray:
        """Euclidean distance from vocab row `idx` to all of V (memoised, BLAS matvec)."""
        d = self._dist_cache.get(idx)
        if d is None:
            M = self.ve.matrix
            d2 = self._sqnorm[idx] + self._sqnorm - 2.0 * (M @ M[idx])
            d = np.sqrt(np.maximum(d2, 0.0))
            self._dist_cache[idx] = d
        return d

    def candidates(self, idx: int, epsilon: float, rng: np.random.Generator):
        """Random adjacency list C_r for vocab row `idx`: (cand_indices, probs, radius).

        The shared core of the mechanism — used by both perturbation and diagnostics so
        measurements reflect the real sampling distribution. `cand`/`probs` are None when
        the radius captures nothing.
        """
        # Per-dimension Laplace scale β_d = δ_d / Z(ε); radius = ‖noise‖ over all dims.
        # (Reference: beta_values = delta_f_new / Z, noise drawn per-dim — func.py.)
        scale = self._delta / self._noise_fn(epsilon)  # [dim] vector
        radius = float(np.linalg.norm(rng.laplace(0.0, scale)))  # random radius = ‖noise‖
        dists = self._dists(idx)  # C_e: cached distance to all V
        cand = np.flatnonzero(dists < radius)  # random adjacency list C_r
        if cand.size == 0:
            return None, None, radius
        u = 1.0 - dists[cand] / radius  # scoring function ∈ (0,1]
        logits = (epsilon / 2.0) * u
        p = np.exp(logits - logits.max())
        p /= p.sum()
        return cand, p, radius

    def _perturb_token(self, surface: str, epsilon: float, rng: np.random.Generator) -> str | None:
        idx = self.ve.index.get(surface)
        if idx is None:
            # out-of-vocab: numbers → random number; everything else → discard.
            return str(rng.integers(1, 1001)) if surface.isdigit() else None
        cand, p, _ = self.candidates(idx, epsilon, rng)
        if cand is None:
            return surface
        return self.ve.vocab[int(rng.choice(cand, p=p))]

    def perturb_aligned(self, text: str, epsilon: float = 3.0, *,
                        seed: int | None = None) -> list[tuple[str, str | None]]:
        """Per-surface (raw, replacement) pairs; replacement is None when discarded.
        Used by attacks that need the raw↔perturbed token alignment."""
        rng = np.random.default_rng(seed)
        return [(s, self._perturb_token(s, epsilon, rng)) for s in self.tok.surfaces(text)]

    def perturb(self, text: str, epsilon: float = 3.0, *, seed: int | None = None) -> str:
        """Return the perturbed document Doc_p for `text` at privacy budget ε."""
        return " ".join(
            r for _, r in self.perturb_aligned(text, epsilon, seed=seed) if r is not None
        )


if __name__ == "__main__":
    # Offline self-check on a tiny synthetic vocab — no network.
    rng = np.random.default_rng(0)
    words = ["year", "day", "month", "hope", "dream", "fear", "city", "town"]
    mat = rng.standard_normal((len(words), 16)).astype(np.float32)
    mat /= np.linalg.norm(mat, axis=1, keepdims=True)
    ve = VocabEmbeddings(words, mat)

    class _FakeTok:
        def surfaces(self, t):
            return t.split()

    p = Perturber(ve, _FakeTok())
    # high ε → small radius → often self/near; low ε → large radius → more change.
    out_hi = p.perturb("year day hope", epsilon=12.0, seed=1)
    out_lo = p.perturb("year day hope", epsilon=0.5, seed=1)
    assert all(w in words for w in out_hi.split()), out_hi
    assert p.perturb("42", seed=1).isdigit(), "numeric token should map to a number"
    print("ε=12:", out_hi, "| ε=0.5:", out_lo)

    # calibrate_noise_fn: realized |C_r|/|V| should land near the target on a fresh embedding.
    big = rng.standard_normal((400, 32)).astype(np.float32)
    big /= np.linalg.norm(big, axis=1, keepdims=True)
    bve = VocabEmbeddings([str(i) for i in range(400)], big)
    nf = calibrate_noise_fn(bve, target=0.05, ref_tokens=[str(i) for i in range(20)])
    bp = Perturber(bve, _FakeTok(), noise_fn=nf)
    fr = np.mean([(lambda c: 0 if c is None else len(c))(bp.candidates(int(t), 6.0, rng)[0]) / 400
                  for t in rng.integers(0, 400, 200)])
    assert 0.02 < fr < 0.10, f"calibrated |C_r| {fr:.3f} not near target 0.05"
    print(f"calibrated |C_r|≈{fr:.3f} (target 0.05)")
    print("OK")
