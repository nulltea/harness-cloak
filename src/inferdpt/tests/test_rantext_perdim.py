"""Regression test for the per-dimension Δφ fix (the word-salad root cause).

The bug: Perturber collapsed the per-dimension sensitivity vector δ_d into a single
scalar δ_max and applied it to every dimension. That inflates the random radius by
√dim·δ_max / ‖δ‖ = δ_max / RMS(δ), pushing it past the bounded unit-norm distance
cloud → |C_r| saturates to ~100% → uniform sampling → word salad.

These tests pin the radius to the per-dim prediction and assert |C_r| stays far from
saturation on a deliberately anisotropic vocab (one inflated dimension) where the
scalar bug would blow up.
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from inferdpt.embeddings import VocabEmbeddings  # noqa: E402
from inferdpt.rantext import Perturber, noise_factor  # noqa: E402


class _NoTok:
    def surfaces(self, t):
        return t.split()


def _anisotropic_vocab(n=2000, dim=128, seed=0):
    """Unit-norm vocab whose dim-0 range is ~10x the others → δ_max ≫ RMS(δ)."""
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, dim)).astype(np.float32)
    M[:, 0] *= 10.0  # blow up one dimension's spread
    M /= np.linalg.norm(M, axis=1, keepdims=True)
    return VocabEmbeddings([f"w{i}" for i in range(n)], M)


def test_radius_is_per_dimension_not_scalar_max():
    """Mean radius must match the per-dim prediction √(2·Σ(δ_d/Z)²), NOT the scalar-max
    prediction √(2·dim·(δ_max/Z)²) which is δ_max/RMS(δ) times larger."""
    ve = _anisotropic_vocab()
    p = Perturber(ve, _NoTok())
    eps = 3.0
    Z = noise_factor(eps)
    delta = ve.sensitivity  # per-dim vector
    dim = ve.matrix.shape[1]
    pred_perdim = math.sqrt(2.0 * np.sum((delta / Z) ** 2))
    pred_scalar = math.sqrt(2.0 * dim * (float(delta.max()) / Z) ** 2)

    rng = np.random.default_rng(1)
    radii = [p.candidates(int(i), eps, rng)[2] for i in rng.integers(0, len(ve.vocab), 400)]
    mean_r = float(np.mean(radii))

    assert pred_scalar > 2.5 * pred_perdim, "test vocab not anisotropic enough to discriminate"
    assert abs(mean_r - pred_perdim) / pred_perdim < 0.1, (mean_r, pred_perdim)
    assert mean_r < 0.5 * pred_scalar, "radius looks like the scalar-max bug"


def test_cr_not_saturated():
    """|C_r|/V must stay far below 100% at ε=3 (the bug gave ~1.0)."""
    ve = _anisotropic_vocab()
    p = Perturber(ve, _NoTok())
    V = len(ve.vocab)
    rng = np.random.default_rng(2)
    fracs = []
    for i in rng.integers(0, V, 300):
        cand, _, _ = p.candidates(int(i), 3.0, rng)
        fracs.append(0.0 if cand is None else len(cand) / V)
    assert np.mean(fracs) < 0.5, f"|C_r|/V saturated: {np.mean(fracs):.3f}"


if __name__ == "__main__":
    test_radius_is_per_dimension_not_scalar_max()
    test_cr_not_saturated()
    print("OK")
