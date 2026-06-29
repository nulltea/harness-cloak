"""Leakage probes — how much of the raw document survives into Doc_p (↓ better)."""

from __future__ import annotations

import numpy as np

from inferdpt.probes._common import content_words, detect_pii, pii_containment


def overlap(doc: str, doc_p: str) -> float:
    """Fraction of the raw doc's content words appearing verbatim in Doc_p. ↓ better."""
    cw = content_words(doc)
    return len(cw & content_words(doc_p)) / (len(cw) or 1)


def s_w_n_w(perturber, words: list[str], epsilon: float, *, runs: int = 100) -> dict[str, float]:
    """Feyisetan plausible-deniability statistics, averaged over `words`:
    S_w = P[M(w)=w] (self-substitution, ↓ better); N_w = #distinct outputs (↑ better).
    Each word is perturbed `runs` times with fresh randomness."""
    s_vals, n_vals = [], []
    for w in words:
        outs = [perturber.perturb(w, epsilon) for _ in range(runs)]  # seed=None → random
        s_vals.append(sum(o.strip().lower() == w.lower() for o in outs) / runs)
        n_vals.append(len(set(outs)))
    return {"S_w": float(np.mean(s_vals)), "N_w": float(np.mean(n_vals))}


def pii_leakage(doc: str, doc_p: str, *, threshold: int = 85,
                entities: list[str] | None = None) -> dict[str, float]:
    """Verbatim/fuzzy survival of raw PII spans into Doc_p (Presidio + rapidfuzz). ↓ better.
    Fuzzy containment, not embedding cosine: cosine floors ~0.7 on bare entities and fails
    multi-word/numeric ones; record-linkage fuzzy match is the reliable containment test."""
    ents = detect_pii(doc) if entities is None else entities
    return pii_containment(ents, doc_p, threshold=threshold)


pii_semantic_leakage = pii_leakage  # back-compat alias
