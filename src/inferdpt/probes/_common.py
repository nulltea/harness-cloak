"""Shared helpers for leakage/utility probes: text scorer, content words, and the
Presidio-based PII matching primitive used by both leakage and utility PII probes."""

from __future__ import annotations

import re
from typing import Callable

import numpy as np

from inferdpt.embeddings import embed  # fixed qwen3-embedding scorer (re-exported)

Embed = Callable[[list[str]], np.ndarray]

_STOP = set("the a an and or of to in for on at is are was were be been has have had he she "
            "it they his her their that this with as by from".split())

# Curated PII entity types (Presidio names). None ⇒ all default recognizers.
PII_ENTITIES = ["PERSON", "LOCATION", "ORGANIZATION", "NRP", "DATE_TIME",
                "EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD", "IBAN_CODE", "US_SSN"]


def content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", text.lower()) if len(w) > 2 and w not in _STOP}


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


_ANALYZER = None


def _analyzer():
    global _ANALYZER
    if _ANALYZER is None:
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NlpEngineProvider

        nlp = NlpEngineProvider(nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
        }).create_engine()
        _ANALYZER = AnalyzerEngine(nlp_engine=nlp)
    return _ANALYZER


def detect_pii(text: str, entities: list[str] | None = PII_ENTITIES) -> list[str]:
    """Return de-duplicated PII surface spans detected in `text`."""
    results = _analyzer().analyze(text=text, language="en", entities=entities)
    seen, spans = set(), []
    for r in results:
        s = text[r.start:r.end].strip()
        if s and s.lower() not in seen:
            seen.add(s.lower())
            spans.append(s)
    return spans


def _candidates(target: str) -> list[str]:
    """Target spans an entity could survive as: content words + consecutive bigrams."""
    toks = re.findall(r"[A-Za-z]+", target)
    uni = [t for t in toks if len(t) > 2]
    bi = [f"{toks[i]} {toks[i+1]}" for i in range(len(toks) - 1)]
    return list(dict.fromkeys(uni + bi)) or [target or " "]


def pii_match_scores(entities: list[str], target: str, *, embed: Embed = embed,
                     tau: float = 0.8) -> dict[str, float]:
    """For each raw PII entity, max cosine to any candidate span in `target`.
    Returns degree (mean max-cosine) and recall (fraction ≥ τ). NaN if no PII."""
    if not entities:
        return {"degree": float("nan"), "recall": float("nan"), "n": 0}
    cands = _candidates(target)
    vecs = embed(entities + cands)
    ent, cand = vecs[:len(entities)], vecs[len(entities):]
    sims = ent @ cand.T  # unit-norm embeddings → dot == cosine
    best = sims.max(axis=1)
    return {"degree": float(best.mean()), "recall": float((best >= tau).mean()), "n": len(entities)}


# ── SimCSE: STS-calibrated cosine scorer (contrastively trained → usable cosine range) ──
SIMCSE_MODEL = "princeton-nlp/sup-simcse-roberta-large"
_SIMCSE = None


def _simcse():
    global _SIMCSE
    if _SIMCSE is None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        torch.set_num_threads(__import__("os").cpu_count() or 4)
        tok = AutoTokenizer.from_pretrained(SIMCSE_MODEL)
        mdl = AutoModel.from_pretrained(SIMCSE_MODEL).eval()
        _SIMCSE = (tok, mdl, torch)
    return _SIMCSE


def simcse_embed(texts: list[str]) -> np.ndarray:
    """SimCSE sentence embeddings (CLS pooling, unit-normalised) — the utility cosine scorer."""
    tok, mdl, torch = _simcse()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 16):
            enc = tok(texts[i:i + 16], padding=True, truncation=True, max_length=256,
                      return_tensors="pt")
            out.append(mdl(**enc).last_hidden_state[:, 0].cpu().numpy())  # [CLS]
    M = np.concatenate(out, 0).astype(np.float32)
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-8)


# ── Cross-encoder reranker: pairwise relevance, via llama-swap /v1/rerank ──
RERANK_URL = "http://localhost:8060/v1/rerank"
RERANK_MODEL = "qwen3-reranker-0.6b"


def rerank(query: str, documents: list[str], *, model: str = RERANK_MODEL,
           url: str = RERANK_URL) -> list[float]:
    """Relevance score in [0,1] of each document to the query (cross-encoder)."""
    import requests

    r = requests.post(url, json={"model": model, "query": query, "documents": documents}, timeout=60)
    r.raise_for_status()
    scores = [0.0] * len(documents)
    for item in r.json()["results"]:
        scores[item["index"]] = float(item["relevance_score"])
    return scores


def relevance(query: str, doc: str) -> float:
    return rerank(query, [doc])[0]


def pii_relevance(entities: list[str], target: str, *, tau: float = 0.8) -> dict[str, float]:
    """Per-entity cross-encoder relevance of each raw PII span to the target text.
    Note: a reranker scores topical relevance, NOT entity containment (reads ~0 for a bare
    entity query even when present), so this is unused for PII; see pii_containment."""
    if not entities:
        return {"degree": float("nan"), "recall": float("nan"), "n": 0}
    s = np.array([relevance(e, target) for e in entities])
    return {"degree": float(s.mean()), "recall": float((s >= tau).mean()), "n": len(entities)}


# ── PII containment via fuzzy string match (record-linkage), the right primitive for
#    "did this entity survive in the text" — no embedding floor; handles verbatim + variants.
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def pii_containment(entities: list[str], target: str, *, threshold: int = 85) -> dict[str, float]:
    """Verbatim/fuzzy survival of each raw PII span in `target` (rapidfuzz). degree = mean
    best ratio in [0,1]; recall = fraction at or above `threshold`. The matcher for both
    PII leakage (on Doc_p) and PII reconstruction (on the output)."""
    from rapidfuzz import fuzz

    if not entities:
        return {"degree": float("nan"), "recall": float("nan"), "n": 0}
    tn = _norm(target)
    scores = []
    for e in entities:
        en = _norm(e)
        s = 100.0 if en and en in tn else max(fuzz.partial_ratio(en, tn), fuzz.token_set_ratio(en, tn))
        scores.append(s / 100.0)
    s = np.array(scores)
    return {"degree": float(s.mean()), "recall": float((s >= threshold / 100.0).mean()), "n": len(entities)}
