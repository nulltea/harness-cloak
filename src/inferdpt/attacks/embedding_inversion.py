"""Embedding-inversion attack (InferDPT-style, token-level).

Threat model: the adversary knows the DP algorithm and φ, so inverts in OUR embedding
cache. For each perturbed token it returns the top-K nearest vocabulary tokens; a hit
means the raw token is among them. recovery@K ↑ helps the attacker (privacy = 1−recovery).
vec2text is out of scope: text — not embedding vectors — is on the wire.
"""

from __future__ import annotations

import numpy as np


def invert(pairs: list[tuple[str, str]], ve, *, ks=(1, 5, 10, 20)) -> dict[str, float]:
    """`pairs` = (raw_word, perturbed_word) for content positions. Both must be in V.

    Batched: all perturbed indices share ONE distance matmul `M @ M[pis].T` (via the
    ‖a−b‖²=‖a‖²+‖b‖²−2a·b identity), then top-k via `argpartition` (O(V)) instead of a
    per-token full `argsort` (O(V log V)). Same recovery@k; far less CPU / BLAS thrash.
    """
    M, idx = ve.matrix, ve.index
    V = len(M)
    valid = [(idx[r], idx[p]) for r, p in pairs if r in idx and p in idx]
    if not valid:
        return {f"recovery@{k}": float("nan") for k in ks}
    ris = np.array([r for r, _ in valid])
    pis = np.array([p for _, p in valid])
    sq = np.einsum("ij,ij->i", M, M)                       # ‖·‖² for all V
    D = sq[None, :] - 2.0 * (M[pis] @ M.T)                 # [n, V], rank-equiv to squared dist
    kmax = max(ks)
    if kmax >= V:                                          # tiny vocab → just sort
        order = np.argsort(D, axis=1)
    else:
        part = np.argpartition(D, kmax, axis=1)[:, :kmax]  # kmax nearest (unordered)
        order = np.take_along_axis(part, np.argsort(np.take_along_axis(D, part, 1), 1), 1)
    hits = {k: [] for k in ks}
    for n, ri in enumerate(ris):
        topk = order[n]
        for k in ks:
            hits[k].append(1.0 if ri in topk[:k] else 0.0)
    return {f"recovery@{k}": float(np.mean(hits[k])) for k in ks}


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
