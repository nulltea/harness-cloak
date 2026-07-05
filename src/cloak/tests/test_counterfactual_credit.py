import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
import train_ranker as tr  # noqa: E402
from test_train_roundtrip_mode import _doc, fake_roundtrip  # noqa: E402


def test_counterfactual_credits_the_flipped_span(monkeypatch):
    monkeypatch.setattr(tr, "roundtrip_batch", fake_roundtrip)
    torch.manual_seed(0)
    doc = _doc()
    policy = tr.RankerPolicy()
    # a rollout that KEPT the level fill (reward 1.0); counterfactual placeholder -> 0.0
    choice, logps, _, doc_p, _ = tr.sample_rollout(doc, doc["spans"], doc["feats"], policy,
                                                   greedy=True)
    if choice["metformin"]["mode"] != "level":   # force the level action for determinism
        choice = {"metformin": doc["spans"][0]["actions"][0]}
        lp = policy.log_probs(doc["feats"][0], doc["spans"][0]["legal"])
        logps = [lp[doc["spans"][0]["legal"].index(0)]]
    term, n_cf = tr.counterfactual_terms(doc, policy, choice, logps, base_r=1.0,
                                         frac=1.0, rng=random.Random(0), rt_workers=1)
    assert n_cf == 1
    # adv_span = base_r - r_cf = 1.0 - 0.0 = 1.0; term = -(adv * logp) > 0
    assert term.item() > 0
    term.backward()   # gradient flows to the policy
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in policy.parameters())


def test_counterfactual_skips_placeholder_spans(monkeypatch):
    monkeypatch.setattr(tr, "roundtrip_batch", fake_roundtrip)
    doc = _doc()
    policy = tr.RankerPolicy()
    ph_action = doc["spans"][0]["actions"][1]
    choice = {"metformin": ph_action}
    lp = policy.log_probs(doc["feats"][0], doc["spans"][0]["legal"])
    logps = [lp[doc["spans"][0]["legal"].index(1)]]
    term, n_cf = tr.counterfactual_terms(doc, policy, choice, logps, base_r=0.0,
                                         frac=1.0, rng=random.Random(0), rt_workers=1)
    assert n_cf == 0 and term == 0.0   # placeholder IS the counterfactual; nothing to flip
