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


def cosine_pairs(pairs: list[tuple[str, str]], *, embed: Embed = simcse_embed) -> list[float]:
    """cos(a,b) for each (a,b) pair, embedding ALL unique texts in ONE batched forward
    (fills the GPU; replaces N tiny 2-text calls). Returns list[float] aligned to `pairs`.
    Use to batch utility/utility_control/u_p_control/coherence across a whole corpus."""
    if not pairs:
        return []
    uniq = list({t for ab in pairs for t in ab})
    V = embed(uniq)
    pos = {t: i for i, t in enumerate(uniq)}
    return [float(cosine(V[pos[a]], V[pos[b]])) for a, b in pairs]


def utility_p_control(gen_control: str, gen_p: str, *, embed: Embed = simcse_embed) -> float:
    """SimCSE cos(non-private remote generation, perturbed remote generation) — REMOTE-output
    utility, measured on Gen_p directly with NO extraction model in the loop. This is the probe
    the paper/reference lack: it catches when Doc_p is word salad → Gen_p drifts off the ideal
    generation, even though the local extractor (which sees the true Doc) still salvages a decent
    final output and masks it in utility/utility_control. ↑ better."""
    v = embed([gen_control, gen_p])
    return cosine(v[0], v[1])


def utility_rerank(doc: str, output: str) -> float:
    """Reranker relevance(Doc, output) — pairwise faithfulness to the raw doc. ↑ better."""
    return relevance(doc, output)


def utility_control_rerank(control_out: str, output: str) -> float:
    """Reranker relevance(non-private generation, output) — fidelity to the ideal. ↑ better."""
    return relevance(control_out, output)


# ── Paper-faithful generation-quality metrics (InferDPT §VI, from SimCTG, Su et al. 2022) ──

def diversity(text: str) -> float:
    """SimCTG diversity = ∏_{n=2}^{4} (unique n-grams / total n-grams) ∈ [0,1]. ↑ better
    (less repetition). Single-text metric — apply to the final output and/or Gen_p. Catches
    degenerate repetition; near-0 for empty/too-short text."""
    toks = text.split()
    d = 1.0
    for n in (2, 3, 4):
        grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
        if not grams:
            return 0.0
        d *= len(set(grams)) / len(grams)
    return d


def coherence(prefix: str, continuation: str, *, embed: Embed = simcse_embed) -> float:
    """SimCTG coherence = cos(SimCSE(prefix), SimCSE(continuation)) — is the continuation
    on-topic with the prefix. ↑ better. Paper applies this to the final output; on Gen_p
    (coherence(doc, gen_p)) it is a clean remote-output utility signal. NB coherence(doc, output)
    is numerically the same as utility(doc, output)."""
    v = embed([prefix, continuation])
    return cosine(v[0], v[1])


def mauve_score(generations: list[str], references: list[str], *, min_n: int = 50,
                featurize_model_name: str = "gpt2-large") -> dict:
    """MAUVE (Pillutla et al. 2021): distribution-level similarity between a set of generations
    and a set of references (here: gold human continuations). ↑ better. CORPUS metric — needs
    many samples to quantize a distribution; returns {'mauve': None, 'skipped': reason} below
    `min_n`. Lazy-imports `mauve` (mauve-text + faiss); featurizes on GPU when available."""
    n = min(len(generations), len(references))
    if n < min_n:
        return {"mauve": None, "n": n, "skipped": f"need ≥{min_n} samples for a valid MAUVE, got {n}"}
    import mauve  # lazy: heavy (faiss + GPT-2 featurizer)
    import torch

    device_id = 0 if torch.cuda.is_available() else -1  # mauve defaults to CPU otherwise
    out = mauve.compute_mauve(p_text=generations[:n], q_text=references[:n], verbose=False,
                              featurize_model_name=featurize_model_name, device_id=device_id)
    return {"mauve": float(out.mauve), "n": n}


def pii_reconstruction_recall(doc: str, output: str, *, threshold: int = 85,
                              entities: list[str] | None = None) -> dict[str, float]:
    """Verbatim/fuzzy recovery of raw PII spans in the final (trusted, local) output
    (Presidio + rapidfuzz containment; see leakage.pii_leakage). ↑ better."""
    ents = detect_pii(doc) if entities is None else entities
    return pii_containment(ents, output, threshold=threshold)
