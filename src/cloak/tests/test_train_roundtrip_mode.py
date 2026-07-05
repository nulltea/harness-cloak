"""Round-trip trainer mode — offline: roundtrip_batch monkeypatched with a deterministic
fake that rewards keeping level fills (so RLOO has a real gradient direction)."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
import train_ranker as tr  # noqa: E402


def _doc():
    actions = [{"mode": "level", "fill": "a biguanide", "aset": 500.0, "p6": 0.8,
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


def fake_roundtrip(jobs, workers=1):
    # reward 1.0 iff the level fill survived into doc_p, else 0.0
    return [{"out_p": "", "out_final": j["doc_p"], "f1s": [float("biguanide" in j["doc_p"])],
             "recall": float("biguanide" in j["doc_p"])} for j in jobs]


def test_sample_rollout_shapes():
    doc = _doc()
    policy = tr.RankerPolicy()
    choice, logps, ph_rate, doc_p, R, legals = tr.sample_rollout(doc, doc["spans"],
                                                                 doc["feats"], policy)
    assert set(choice) == {"metformin"} and len(logps) == 1
    assert isinstance(doc_p, str) and isinstance(R, list)
    assert len(legals) == 1 and legals[0] == doc["spans"][0]["legal"]


def test_rloo_advantage_no_std():
    r = torch.tensor([1.0, 0.0, 0.0, 0.0])
    adv = tr.rloo_advantage(r)
    # b_g = mean of others: adv_0 = 1 - 0 = 1.0; adv_j = 0 - 1/3
    assert torch.allclose(adv, torch.tensor([1.0, -1 / 3, -1 / 3, -1 / 3]))


def test_roundtrip_epoch_moves_policy(monkeypatch):
    monkeypatch.setattr(tr, "roundtrip_batch", fake_roundtrip)
    doc = _doc()
    torch.manual_seed(0)
    policy = tr.RankerPolicy()
    before = policy.log_probs(doc["feats"][0], doc["spans"][0]["legal"]).detach().clone()
    stats = tr.train_roundtrip([doc], policy, G=4, epochs=3, lr=0.05,
                               entropy_coef=0.01, kl_coef=0.0, ref=None,
                               rt_workers=1, seed=0)
    after = policy.log_probs(doc["feats"][0], doc["spans"][0]["legal"]).detach()
    assert not torch.allclose(before, after)          # first-smoke movement canary
    assert after[0] > before[0]                       # level action (rewarded) went UP
    assert "ties_skipped" in stats[-1]
