import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
import train_ranker as tr  # noqa: E402
from test_train_roundtrip_mode import _doc, fake_roundtrip  # noqa: E402


def test_exit_round_selects_winner_and_clones(monkeypatch):
    monkeypatch.setattr(tr, "roundtrip_batch", fake_roundtrip)
    torch.manual_seed(0)
    doc = _doc()
    policy = tr.RankerPolicy()
    winners, stats = tr.exit_round([doc], policy, G=6, rt_workers=1, seed=0)
    # fake reward pays 1.0 for the level fill; the winner must have chosen action 0
    assert stats["n_winners"] == 1 and stats["mean_best_r"] == 1.0
    (di, choice_idx), = winners
    assert di == 0 and choice_idx["metformin"] == 0
    before = policy.log_probs(doc["feats"][0], doc["spans"][0]["legal"]).detach().clone()
    tr.clone_choices(policy, [(doc["spans"], doc["feats"], choice_idx)], epochs=20, lr=0.05)
    after = policy.log_probs(doc["feats"][0], doc["spans"][0]["legal"]).detach()
    assert after[0] > before[0]     # SFT on the winner raises the winning action's logp


def test_exit_round_no_winner_when_bc_optimal(monkeypatch):
    # reward everything 0 -> nothing beats the BC baseline -> no winners
    monkeypatch.setattr(tr, "roundtrip_batch",
                        lambda jobs, workers=1: [{"out_p": "", "out_final": "", "f1s": [0.0],
                                                  "recall": 0.0} for _ in jobs])
    torch.manual_seed(0)
    doc = _doc()
    winners, stats = tr.exit_round([doc], tr.RankerPolicy(), G=4, rt_workers=1, seed=0)
    assert winners == [] and stats["n_winners"] == 0
