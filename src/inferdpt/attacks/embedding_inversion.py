"""Embedding-inversion attack (InferDPT-style, token-level).

Threat model: the adversary knows the DP algorithm and φ, so inverts in OUR embedding
cache. For each perturbed token it returns the top-K nearest vocabulary tokens; a hit
means the raw token is among them. recovery@K ↑ helps the attacker (privacy = 1−recovery).
vec2text is out of scope: text — not embedding vectors — is on the wire.
"""

from __future__ import annotations

import numpy as np


def invert(pairs: list[tuple[str, str]], ve, *, ks=(1, 5, 10, 20)) -> dict[str, float]:
    """`pairs` = (raw_word, perturbed_word) for content positions. Both must be in V."""
    M, idx = ve.matrix, ve.index
    hits = {k: [] for k in ks}
    for raw, pert in pairs:
        ri, pi = idx.get(raw), idx.get(pert)
        if ri is None or pi is None:
            continue
        order = np.argsort(np.linalg.norm(M - M[pi], axis=1))  # nearest first (incl. pert@0)
        for k in ks:
            hits[k].append(1.0 if ri in order[:k] else 0.0)
    return {f"recovery@{k}": (float(np.mean(hits[k])) if hits[k] else float("nan")) for k in ks}


if __name__ == "__main__":
    # Offline self-check: a perturbed token whose true raw is its 2nd-nearest neighbour.
    from dataclasses import dataclass

    @dataclass
    class _VE:
        vocab: list
        matrix: np.ndarray
        def __post_init__(self):
            self.index = {w: i for i, w in enumerate(self.vocab)}

    M = np.array([[0, 0], [0.1, 0], [1, 1], [0.2, 0]], dtype=np.float32)  # raw=idx0 near pert=idx1
    ve = _VE(["raw", "pert", "far", "mid"], M)
    r = invert([("raw", "pert")], ve, ks=(1, 2, 4))
    assert r["recovery@1"] == 0.0 and r["recovery@2"] == 1.0 and r["recovery@4"] == 1.0, r
    print("OK", r)
