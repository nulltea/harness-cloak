"""Candidate-sensitive privacy probes for the substitutor (probe-per-job, shootout-validated).

Two probes, two jobs (docs/specs/attacks.md §3; docs/specs/RL/surrogate-ranker-infiller.md §4.2):

- walk_risk (contrastive re-identification): P(attacker picks the original out of a same-type
  anonymity set | context + visible fill). Length-normalized causal-LM log-probs (pythia-410m)
  softmaxed over {original} ∪ sampled same-type corpus distractors. Serves the τ-walk and the
  RL action mask — best within-span level-ordering under both shootout referees (.86/.71).
- fill_proximity / reward_privacy (embedding proximity): cos_MiniLM(fill, original), the RL
  reward's privacy term A — best attacker-discrimination AUC (.83/.77). Context-blind and
  embedding-gameable by a trainable generator: stage-2 requires walk_risk joining A or the
  document-level head (pre-registered guard).

Both replace the legacy mask-away MTI probe, which was candidate-INVARIANT (identical score for
every lattice level of a span — degenerate τ-walk, zero RL privacy gradient; measured
2026-07-03) and the appositive variant, which failed attacker correlation (AUC ≈ chance).

Distractor pools: data/probe_distractors.json ({type: [surfaces]}, built by
scripts/build_probe_distractors.py from the arms artifact). Unpooled span types fail closed
(risk 1.0 → the walk exhausts → placeholder path).
"""
import json
import random
import re
import zlib
from pathlib import Path

CAUSAL_MODEL = "EleutherAI/pythia-410m"
EMBED_MODEL = "all-MiniLM-L6-v2"
POOLS_PATH = Path("data/probe_distractors.json")
N_DISTRACTORS = 15
MIN_POOL = 4

_fill = None
_causal = None
_embed = None
_pools = None


def _pipe():
    """roberta fill-mask — kept for the sweep script's span-inversion attack probe."""
    global _fill
    if _fill is None:
        import torch
        from transformers import pipeline
        _fill = pipeline("fill-mask", model="roberta-base", top_k=50,
                         device=0 if torch.cuda.is_available() else -1)
    return _fill


def _lm():
    global _causal
    if _causal is None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(CAUSAL_MODEL)
        model = AutoModelForCausalLM.from_pretrained(CAUSAL_MODEL, torch_dtype=torch.float16)
        model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
        _causal = (tok, model)
    return _causal


def _load_pools() -> dict:
    global _pools
    if _pools is None:
        _pools = json.loads(POOLS_PATH.read_text()) if POOLS_PATH.exists() else {}
    return _pools


def _logp_continuations(prefix: str, conts: list[str]) -> list[float]:
    """Length-normalized logP(cont | prefix) per continuation, one batched forward."""
    import torch
    tok, model = _lm()
    pre = tok(prefix)["input_ids"]
    seqs = [pre + tok(" " + c, add_special_tokens=False)["input_ids"] for c in conts]
    maxlen = max(len(s) for s in seqs)
    pad = tok.pad_token_id or 0
    batch = torch.tensor([s + [pad] * (maxlen - len(s)) for s in seqs]).to(model.device)
    with torch.no_grad():
        logits = model(batch).logits.log_softmax(-1)
    out = []
    for row, s in enumerate(seqs):
        lp = [logits[row, i - 1, s[i]].item() for i in range(len(pre), len(s))]
        out.append(sum(lp) / len(lp) if lp else -1e9)
    return out


def walk_risk(sent_with_fill: str, original: str, fill: str, span_type: str) -> float:
    """Contrastive re-identification risk in [0,1] for one (span, fill) in its sentence.

    Deterministic (distractor sample seeded by the span) so the walk and the RL risk table
    are reproducible. Fail-closed 1.0 when the type has no distractor pool.
    """
    import torch
    pool = [s for s in _load_pools().get(span_type, [])
            if s.lower() != original.lower()]
    if len(pool) < MIN_POOL:
        return 1.0  # unpooled type: conservative -> exhaustion -> placeholder path
    rng = random.Random(zlib.crc32(f"{original.lower()}|{span_type}".encode()))
    cands = [original] + rng.sample(pool, min(N_DISTRACTORS, len(pool)))
    prefix = f'{sent_with_fill}\nThe anonymized phrase "{fill}" originally read:'
    lps = _logp_continuations(prefix, cands)
    return float(torch.tensor(lps).softmax(-1)[0])


def fill_proximity(fill: str, original: str) -> float:
    """cos_MiniLM(fill, original) — the reward's per-span privacy signal (P6)."""
    global _embed
    if _embed is None:
        from sentence_transformers import SentenceTransformer
        _embed = SentenceTransformer(EMBED_MODEL)
    a, b = _embed.encode([fill, original], normalize_embeddings=True)
    return float((a * b).sum())


def reward_privacy(R: list[dict]) -> dict:
    """A(doc_p, R): mean fill-original proximity over generalized spans (max = diagnostic).

    Generic placeholders are excluded: the label carries no original-specific signal and
    contributes a constant (no gradient). Context is deliberately absent — P6 is context-blind
    by construction; the frontier attacker at eval prices what it misses.
    """
    vals = [fill_proximity(e["replacement"], e["surface"])
            for e in R if e["action"] == "generalize"]
    if not vals:
        return {"mean": 0.0, "max": 0.0, "n": 0}
    return {"mean": sum(vals) / len(vals), "max": max(vals), "n": len(vals)}


if __name__ == "__main__":
    # pools may be freshly built or absent; self-check builds a tiny in-memory pool
    _pools = {"LOC": ["Oslo", "Bergen", "Stockholm", "Copenhagen", "Helsinki", "Reykjavik",
                      "Gothenburg", "Aarhus", "Trondheim", "Malmo"]}
    ctx = "She lives in {} with her family."
    r_spec = walk_risk(ctx.format("a Norwegian city"), "Oslo", "a Norwegian city", "LOC")
    r_coarse = walk_risk(ctx.format("a place"), "Oslo", "a place", "LOC")
    r_unpooled = walk_risk(ctx.format("a place"), "Oslo", "a place", "NOSUCHTYPE")
    print(f"walk_risk 'a Norwegian city'={r_spec:.4f} 'a place'={r_coarse:.4f} "
          f"unpooled={r_unpooled}")
    assert 0 <= r_coarse <= 1 and 0 <= r_spec <= 1
    assert r_spec > r_coarse, (r_spec, r_coarse)      # specificity must cost risk
    assert r_unpooled == 1.0                          # fail closed
    assert walk_risk(ctx.format("a place"), "Oslo", "a place", "LOC") == r_coarse  # deterministic

    p_near = fill_proximity("a Norwegian city", "Oslo")
    p_far = fill_proximity("a place", "Oslo")
    print(f"fill_proximity near={p_near:.4f} far={p_far:.4f}")
    assert p_near > p_far

    R = [{"action": "generalize", "surface": "Oslo", "replacement": "a Norwegian city"},
         {"action": "placeholder", "surface": "Sarah", "replacement": "<PERSON_1>"}]
    a = reward_privacy(R)
    print("reward_privacy:", {k: round(v, 4) if isinstance(v, float) else v for k, v in a.items()})
    assert a["n"] == 1 and abs(a["mean"] - p_near) < 1e-6
    print("probe.py self-check OK")
