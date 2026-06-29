"""Utility probes — how faithful the final output is (↑ better).

Two scorer families (pick per report): SimCSE cosine (STS-calibrated, paper-faithful,
usable dynamic range) and a cross-encoder reranker relevance score (more discriminative,
free of the cosine-geometry problem). PII reconstruction uses the reranker (entity-in-text
relevance, better than sentence cosine on short spans).
"""

from __future__ import annotations

from inferdpt.probes._common import Embed, cosine, detect_pii, pii_containment, relevance, simcse_embed


def utility(doc: str, output: str, *, embed: Embed = simcse_embed) -> float:
    """SimCSE cos(Doc, output) — output on-topic with the raw doc. ↑ better."""
    v = embed([doc, output])
    return cosine(v[0], v[1])


def utility_control(control_out: str, output: str, *, embed: Embed = simcse_embed) -> float:
    """SimCSE cos(non-private generation, output) — fidelity to the ideal answer. ↑ better."""
    v = embed([control_out, output])
    return cosine(v[0], v[1])


def utility_rerank(doc: str, output: str) -> float:
    """Reranker relevance(Doc, output) — pairwise faithfulness to the raw doc. ↑ better."""
    return relevance(doc, output)


def utility_control_rerank(control_out: str, output: str) -> float:
    """Reranker relevance(non-private generation, output) — fidelity to the ideal. ↑ better."""
    return relevance(control_out, output)


def pii_reconstruction_recall(doc: str, output: str, *, threshold: int = 85,
                              entities: list[str] | None = None) -> dict[str, float]:
    """Verbatim/fuzzy recovery of raw PII spans in the final (trusted, local) output
    (Presidio + rapidfuzz containment; see leakage.pii_leakage). ↑ better."""
    ents = detect_pii(doc) if entities is None else entities
    return pii_containment(ents, output, threshold=threshold)
