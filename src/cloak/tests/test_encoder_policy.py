"""EncoderPolicy contract — uses a tiny random HF encoder so the test stays fast/offline
(hf-internal-testing models are ~100KB and live in the shared HF cache)."""
import torch

from cloak.train.ranker import N_FEAT, EncoderPolicy, span_context

TINY = "hf-internal-testing/tiny-random-bert"


def test_span_context_windows():
    text = "A" * 300 + " metformin " + "B" * 300
    ctx = span_context(text, start=301, window=50)
    assert "metformin" in ctx and len(ctx) <= 120


def test_encoder_policy_contract():
    torch.manual_seed(0)
    pol = EncoderPolicy(encoder_name=TINY)
    ctx = pol.embed_contexts(["patient on metformin 500mg daily"])
    assert ctx.shape[0] == 1
    pol.set_context(ctx[0])
    feats = torch.randn(3, N_FEAT)
    legal = [0, 2]
    lp = pol.log_probs(feats, legal)
    assert lp.shape == (2,) and torch.isfinite(lp).all()
    assert abs(lp.exp().sum().item() - 1.0) < 1e-4
    a, alp = pol.sample(feats, legal, greedy=True)
    assert a in legal and torch.isfinite(alp)
    # frozen encoder: only head params require grad
    enc_trainable = [p for p in pol.encoder.parameters() if p.requires_grad]
    assert not enc_trainable
