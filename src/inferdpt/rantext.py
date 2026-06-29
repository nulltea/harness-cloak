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


class Perturber:
    def __init__(
        self,
        vocab: VocabEmbeddings,
        tokeniser: Tokeniser | None = None,
        *,
        noise_scale: float = 0.38,
    ) -> None:
        self.ve = vocab
        self.tok = tokeniser or Tokeniser()
        self._delta = vocab.sensitivity  # Δφ
        # Distance from a token to all of V is independent of noise and ε, so cache it.
        # dist² = ‖a‖² + ‖b‖² − 2a·b → one BLAS matvec (multicore) per token, then memoised.
        self._sqnorm = np.einsum("ij,ij->i", vocab.matrix, vocab.matrix)
        self._dist_cache: dict[int, np.ndarray] = {}
        # ponytail: calibration knob for the embedding geometry. The paper's Z(ε)+Δφ
        # were tuned for ada-002; unit-norm qwen3 vectors have concentrated distances,
        # so without this the radius engulfs all of V. Upgrade path: cosine-based radius
        # or a less-concentrated embedding if neighbourhoods stay brittle.
        self.noise_scale = noise_scale

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
        dim = self.ve.matrix.shape[1]
        scale = self.noise_scale * self._delta / noise_factor(epsilon)
        radius = float(np.linalg.norm(rng.laplace(0.0, scale, size=dim)))  # random radius = ‖noise‖
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
    print("OK")
