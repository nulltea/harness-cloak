"""MTI guess-back probe (masked-token inference, roberta-base).

Selection-time attack: mask the replacement span in context; risk = the probability
mass the MLM puts on the original's content tokens. The lattice walk accepts the
first level with risk < tau. After sjmeis/EpsilonDistributor's MaskedTokenInference
(Yue et al. 2106.01221 lineage), adapted token->span.

Semantics note: masking hides the candidate, so risk = P(original | context minus span)
— candidate-INVARIANT within a slot. It is a span-level "contextually inferable anyway"
detector: if risk >= tau at every lattice level, the walk takes the most-general level
and records the risk in R. ponytail: candidate-sensitive ranking (guess-back given the
substituted text, e.g. pseudo-likelihood or LLM probe) is a P2 refinement.
"""
import re

_fill = None
MODEL = "roberta-base"


def _pipe():
    global _fill
    if _fill is None:
        import torch
        from transformers import pipeline
        _fill = pipeline("fill-mask", model=MODEL, top_k=50,
                         device=0 if torch.cuda.is_available() else -1)
    return _fill


def guess_back_risk(context: str, original: str, replacement: str) -> float:
    """Mask `replacement` inside `context` (which contains it); return the summed
    probability of the original's content tokens among top-50 single-mask predictions."""
    fill = _pipe()
    pat = re.compile(re.escape(replacement), re.IGNORECASE)
    if not pat.search(context):
        return 1.0  # can't probe -> treat as maximally risky (fail closed)
    masked = pat.sub(fill.tokenizer.mask_token, context, count=1)
    preds = fill(masked)  # contexts are single sentences; no truncation kwarg in 5.x
    targets = {t.lower() for t in re.findall(r"\w+", original) if len(t) > 2} or {original.lower()}
    return sum(p["score"] for p in preds if p["token_str"].strip().lower() in targets)


if __name__ == "__main__":
    ctx = "She works as a cardiologist at the hospital in Oslo."
    r_same = guess_back_risk(ctx, "cardiologist", "cardiologist")
    r_gen = guess_back_risk(ctx.replace("cardiologist", "specialist"), "cardiologist", "specialist")
    r_far = guess_back_risk(ctx.replace("cardiologist", "professional"), "cardiologist", "professional")
    print(f"risk self={r_same:.4f} specialist={r_gen:.4f} professional={r_far:.4f}")
    assert r_same >= r_gen >= 0 and r_same >= r_far >= 0
    print("probe.py self-check OK")
