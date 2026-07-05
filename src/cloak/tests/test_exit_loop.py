import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
import train_ranker as tr  # noqa: E402
from test_train_roundtrip_mode import _doc, fake_roundtrip  # noqa: E402


def _doc_exit():
    """Like _doc but with a floor-walk baseline that scores 0, so a rollout can beat it.
    Three actions: idx0 level 'a generic agent' (aset 500, NOT rewarded), idx1 level
    'a biguanide' (aset 1000, rewarded), idx2 placeholder. Floor-walk = min-aset legal
    level = idx0, whose fill lacks 'biguanide' → baseline round-trip reward 0.0."""
    actions = [{"mode": "level", "fill": "a generic agent", "aset": 500.0, "p6": 0.5,
                "walk_risk": 0.0},
               {"mode": "level", "fill": "a biguanide", "aset": 1000.0, "p6": 0.8,
                "walk_risk": 0.0},
               {"mode": "placeholder", "fill": "<QUANTITY_1>", "p6": 0.0, "walk_risk": 0.0}]
    span = {"surface": "metformin", "type": "QUANTITY", "start": 0, "actions": actions}
    raw = [dict(span)]
    spans, feats = tr.derive_spans(raw, {"QUANTITY": 100.0}, "clinical", "cpu")
    return {"id": "d0", "corpus": "clinical", "text": "metformin daily",
            "R_walk": [{"surface": "metformin", "type": "QUANTITY", "action": "generalize",
                        "replacement": "a biguanide", "start": 0, "end": 9,
                        "lattice": ["a biguanide"]}],
            "raw_spans": raw, "spans": spans, "feats": feats,
            "probes_train": [{"surface": "metformin", "question": "What drug?"}]}


def test_exit_round_selects_winner_and_clones(monkeypatch):
    monkeypatch.setattr(tr, "roundtrip_batch", fake_roundtrip)
    torch.manual_seed(0)
    doc = _doc_exit()
    assert doc["spans"][0]["bc_action"] == 0   # floor-walk = min-aset legal level (aset 500)
    policy = tr.RankerPolicy()
    winners, stats = tr.exit_round([doc], policy, G=12, rt_workers=1, seed=0)
    # floor-walk baseline ('a generic agent') scores 0; only the 'a biguanide' rollout wins
    assert stats["n_winners"] == 1 and stats["mean_bc_r"] == 0.0
    (di, choice_idx), = winners
    assert di == 0 and choice_idx["metformin"] == 1
    legal = doc["spans"][0]["legal"]
    before = policy.log_probs(doc["feats"][0], legal).detach().clone()
    tr.clone_choices(policy, [(doc["spans"], doc["feats"], choice_idx)], epochs=20, lr=0.05)
    after = policy.log_probs(doc["feats"][0], legal).detach()
    assert after[legal.index(1)] > before[legal.index(1)]   # SFT raised the winner's logp


def test_exit_round_no_winner_when_bc_optimal(monkeypatch):
    # reward everything 0 -> nothing beats the BC baseline -> no winners
    monkeypatch.setattr(tr, "roundtrip_batch",
                        lambda jobs, workers=1: [{"out_p": "", "out_final": "", "f1s": [0.0],
                                                  "recall": 0.0} for _ in jobs])
    torch.manual_seed(0)
    doc = _doc()
    winners, stats = tr.exit_round([doc], tr.RankerPolicy(), G=4, rt_workers=1, seed=0)
    assert winners == [] and stats["n_winners"] == 0
