"""Mask/BERT inference attack (SanText/InferDPT §VI-B).

Slides [MASK] across each perturbed content-word position and lets a pretrained BERT
MLM predict the raw word from the perturbed context. Reports a binary recovery rate
plus a GRADED leakage signal (posterior mass on / rank of the true token) — the graded
part is well-defined only for raw words that are a single BERT wordpiece.

Word-level single-[MASK]: a raw word BERT would split into >1 wordpiece can't be matched
by a single prediction → counted as not-recovered (conservative, over-estimates privacy).
"""

from __future__ import annotations

import numpy as np

_TOK = _MDL = _TORCH = _DEVICE = None


def _load(model: str):
    global _TOK, _MDL, _TORCH, _DEVICE
    if _MDL is None:
        import os

        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        if _DEVICE == "cpu":
            torch.set_num_threads(os.cpu_count() or 4)  # CPU fallback: use all cores
        _TORCH = torch
        _TOK = AutoTokenizer.from_pretrained(model)
        _MDL = AutoModelForMaskedLM.from_pretrained(model).eval().to(_DEVICE)
    return _TOK, _MDL, _TORCH, _DEVICE


def reconstruct(perturbed_words: list[str], raw_words: list[str], is_content,
                *, model: str = "bert-base-uncased", batch: int = 64) -> dict[str, float]:
    """`perturbed_words` = the Doc_p word sequence; `raw_words` = aligned raw words.
    `is_content(word)->bool` selects which positions to attack.

    All masked sentences (one per attacked position) run as a SINGLE batched GPU forward
    pass instead of one CPU forward per token — the old per-token CPU loop was the eval's
    dominant CPU sink (all-core BERT × tokens × docs × ε)."""
    tok, mdl, torch, device = _load(model)
    mask = tok.mask_token
    sents = [(" ".join(w if j != i else mask for j, w in enumerate(perturbed_words)), raw)
             for i, raw in enumerate(raw_words) if is_content(raw)]
    top1, top5, post, ranks = [], [], [], []
    for b0 in range(0, len(sents), batch):
        chunk = sents[b0:b0 + batch]
        enc = tok([s for s, _ in chunk], return_tensors="pt", padding=True,
                  truncation=True, max_length=256).to(device)
        with torch.no_grad():
            logits = mdl(**enc).logits  # [B, T, vocab]
        for bi, (_, raw) in enumerate(chunk):
            pos = (enc.input_ids[bi] == tok.mask_token_id).nonzero(as_tuple=True)[0]
            if len(pos) != 1:  # word re-tokenised oddly / mask truncated → skip
                continue
            probs = logits[bi, pos[0]].softmax(-1)
            preds = [tok.decode([t]).strip().lower() for t in probs.topk(5).indices.tolist()]
            top1.append(float(preds[0] == raw.lower()))
            top5.append(float(raw.lower() in preds))
            rid = tok(raw, add_special_tokens=False).input_ids  # graded: single-wordpiece only
            if len(rid) == 1:
                post.append(float(probs[rid[0]]))
                ranks.append(int((probs > probs[rid[0]]).sum()) + 1)

    m = lambda xs: float(np.mean(xs)) if xs else float("nan")
    return {"top1_recovery": m(top1), "top5_recovery": m(top5),
            "mean_posterior_true": m(post), "mean_rank_true": m(ranks), "n": len(top1)}
