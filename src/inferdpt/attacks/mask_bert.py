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

_TOK = _MDL = _TORCH = None


def _load(model: str):
    global _TOK, _MDL, _TORCH
    if _MDL is None:
        import os

        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        torch.set_num_threads(os.cpu_count() or 4)  # CPU BERT: use all cores
        _TORCH = torch
        _TOK = AutoTokenizer.from_pretrained(model)
        _MDL = AutoModelForMaskedLM.from_pretrained(model).eval()
    return _TOK, _MDL, _TORCH


def reconstruct(perturbed_words: list[str], raw_words: list[str], is_content,
                *, model: str = "bert-base-uncased") -> dict[str, float]:
    """`perturbed_words` = the Doc_p word sequence; `raw_words` = aligned raw words.
    `is_content(word)->bool` selects which positions to attack."""
    tok, mdl, torch = _load(model)
    mask = tok.mask_token
    top1, top5, post, ranks = [], [], [], []
    for i, raw in enumerate(raw_words):
        if not is_content(raw):
            continue
        sent = list(perturbed_words)
        sent[i] = mask
        enc = tok(" ".join(sent), return_tensors="pt")
        pos = (enc.input_ids[0] == tok.mask_token_id).nonzero(as_tuple=True)[0]
        if len(pos) != 1:  # word re-tokenised oddly → skip
            continue
        with torch.no_grad():
            probs = mdl(**enc).logits[0, pos[0]].softmax(-1)
        topids = probs.topk(5).indices.tolist()
        preds = [tok.decode([t]).strip().lower() for t in topids]
        top1.append(float(preds[0] == raw.lower()))
        top5.append(float(raw.lower() in preds))
        rid = tok(raw, add_special_tokens=False).input_ids  # graded: single-wordpiece only
        if len(rid) == 1:
            p = float(probs[rid[0]])
            post.append(p)
            ranks.append(int((probs > probs[rid[0]]).sum()) + 1)

    m = lambda xs: float(np.mean(xs)) if xs else float("nan")
    return {"top1_recovery": m(top1), "top5_recovery": m(top5),
            "mean_posterior_true": m(post), "mean_rank_true": m(ranks), "n": len(top1)}
