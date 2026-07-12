"""Round-trip trainer mode — offline: roundtrip_batch monkeypatched with a deterministic
fake that rewards keeping level fills (so RLOO has a real gradient direction)."""
import sys
from pathlib import Path

import pytest
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


def test_inert_floors_make_all_actions_legal_but_bc_skips_keep_original():
    actions = [
        {"mode": "level", "fill": "a narrow clinical drug class", "aset": 17.0, "p6": 0.6,
         "walk_risk": 0.0},
        {"mode": "level", "fill": "a broad medication", "aset": 250.0, "p6": 0.4,
         "walk_risk": 0.0},
        {"mode": "level", "fill": "metformin", "keep": True, "aset": 1.0, "p6": 1.0,
         "walk_risk": 1.0},
        {"mode": "placeholder", "fill": "<DRUG_1>", "p6": 0.0, "walk_risk": 0.0},
    ]
    raw = [{"surface": "metformin", "type": "drug", "start": 0, "actions": actions}]

    spans, _feats = tr.derive_spans(raw, {"drug": 1.0, "OTHER": 1.0}, "clinical", "cpu")
    span = spans[0]

    assert span["legal"] == [0, 1, 2, 3]
    assert span["bc_action"] == 0
    assert not span["actions"][span["bc_action"]].get("keep")
    assert tr.floor_walk_choice(spans)["metformin"]["fill"] == "a narrow clinical drug class"


def _exit_doc():
    actions = [{"mode": "level", "fill": "a biguanide", "aset": 100.0, "p6": 0.8,
                "walk_risk": 0.0},
               {"mode": "level", "fill": "a medication", "aset": 200.0, "p6": 0.7,
                "walk_risk": 0.0},
               {"mode": "placeholder", "fill": "<QUANTITY_1>", "p6": 0.0,
                "walk_risk": 0.0}]
    span = {"surface": "metformin", "type": "QUANTITY", "start": 0, "actions": actions}
    raw = [dict(span)]
    spans, feats = tr.derive_spans(raw, {"QUANTITY": 100.0}, "clinical", "cpu")
    return {"id": "d0", "corpus": "clinical", "text": "metformin daily",
            "R_walk": [{"surface": "metformin", "type": "QUANTITY", "action": "generalize",
                        "replacement": "a biguanide", "start": 0, "end": 9,
                        "lattice": ["a biguanide", "a medication"]}],
            "raw_spans": raw, "spans": spans, "feats": feats,
            "probes_train": [{"surface": "metformin", "question": "What drug?"}]}


def fake_roundtrip(jobs, workers=1, reader_refresh=False):
    # reward 1.0 iff the level fill survived into doc_p, else 0.0
    return [{"out_p": "", "out_final": j["doc_p"], "f1s": [float("biguanide" in j["doc_p"])],
             "recall": float("biguanide" in j["doc_p"])} for j in jobs]


def _with_carrier(doc):
    doc = dict(doc)
    doc["ladder"] = [{"surface": "metformin", "rung": 0, "q": "What drug?",
                      "rungs": ["metformin", "a biguanide"]}]
    doc["decisions"] = [
        {"q": "Which route?", "options": ["endocrinology", "primary care"],
         "gold": "primary care", "span_ids": ["metformin"]},
    ]
    doc["out_hi"] = "ASSESSMENT: metformin - medication - active"
    doc["schema"] = True
    return doc


def _candidate_rollout(doc, span_rows, feats, policy):
    choice = {"metformin": span_rows[0]["actions"][1]}
    doc_p, R = tr.assemble(doc["text"], doc["R_walk"], span_rows, choice)
    return choice, [], 0.0, doc_p, R, []


def _rollout_sequence(action_indices):
    actions = iter(action_indices)

    def rollout(doc, span_rows, feats, policy):
        action_idx = next(actions)
        choice = {"metformin": span_rows[0]["actions"][action_idx]}
        doc_p, R = tr.assemble(doc["text"], doc["R_walk"], span_rows, choice)
        return choice, [], 0.0, doc_p, R, []

    return rollout


def _assert_serial_verify_calls(calls):
    assert len(calls) == 3
    assert calls[0]["workers"] == 9
    assert calls[0]["reader_refresh"] is False
    assert len(calls[0]["doc_ps"]) == 2
    assert all(call["workers"] == 1 for call in calls[1:])
    assert all(call["reader_refresh"] is True for call in calls[1:])
    assert sorted(call["doc_ps"][0] for call in calls[1:]) == [
        "A biguanide daily", "A medication daily"]


def test_exit_round_keeps_candidate_after_serial_reverification(monkeypatch):
    doc = _exit_doc()
    calls = []

    def fake_roundtrip_exit(jobs, workers=1, reader_refresh=False):
        calls.append({"doc_ps": [j["doc_p"] for j in jobs], "workers": workers,
                      "reader_refresh": reader_refresh})
        if not reader_refresh:
            return [{"recall": 0.2}, {"recall": 0.9}]
        return [{"recall": 0.8 if "medication" in jobs[0]["doc_p"].lower() else 0.4}]

    monkeypatch.setattr(tr, "sample_rollout", _candidate_rollout)
    monkeypatch.setattr(tr, "roundtrip_batch", fake_roundtrip_exit)

    winners, stats = tr.exit_round([doc], tr.RankerPolicy(), G=1, rt_workers=9, seed=0)

    assert winners == [(0, {"metformin": 1})]
    assert stats["n_candidates"] == 1
    assert stats["n_winners"] == 1
    assert stats["n_verify_dropped"] == 0
    _assert_serial_verify_calls(calls)


def test_exit_round_drops_candidate_when_clean_baseline_catches_up(monkeypatch):
    doc = _exit_doc()
    calls = []

    def fake_roundtrip_exit(jobs, workers=1, reader_refresh=False):
        calls.append({"doc_ps": [j["doc_p"] for j in jobs], "workers": workers,
                      "reader_refresh": reader_refresh})
        if not reader_refresh:
            return [{"recall": 0.2}, {"recall": 0.9}]
        return [{"recall": 0.3 if "medication" in jobs[0]["doc_p"].lower() else 0.4}]

    monkeypatch.setattr(tr, "sample_rollout", _candidate_rollout)
    monkeypatch.setattr(tr, "roundtrip_batch", fake_roundtrip_exit)

    winners, stats = tr.exit_round([doc], tr.RankerPolicy(), G=1, rt_workers=9, seed=0)

    assert winners == []
    assert stats["n_candidates"] == 1
    assert stats["n_winners"] == 0
    assert stats["n_verify_dropped"] == 1
    _assert_serial_verify_calls(calls)


def test_exit_round_uses_verification_as_tiebreak_and_keeps_one_winner(monkeypatch):
    doc = _exit_doc()
    calls = []

    def fake_roundtrip_exit(jobs, workers=1, reader_refresh=False):
        calls.append({"doc_ps": [j["doc_p"] for j in jobs], "workers": workers,
                      "reader_refresh": reader_refresh})
        if not reader_refresh:
            return [{"recall": 0.2}, {"recall": 0.9}, {"recall": 0.9}]
        doc_p = jobs[0]["doc_p"].lower()
        if "biguanide" in doc_p:
            return [{"recall": 0.4}]
        if "medication" in doc_p:
            return [{"recall": 0.8}]
        return [{"recall": 0.3}]

    monkeypatch.setattr(tr, "sample_rollout", _rollout_sequence([2, 1]))
    monkeypatch.setattr(tr, "roundtrip_batch", fake_roundtrip_exit)

    winners, stats = tr.exit_round([doc], tr.RankerPolicy(), G=2, rt_workers=9, seed=0)

    assert winners == [(0, {"metformin": 1})]
    assert stats["n_candidates"] == 1
    assert stats["n_winners"] == 1
    assert stats["n_verify_dropped"] == 0
    assert stats["n_candidates"] == stats["n_winners"] + stats["n_verify_dropped"]
    assert len(calls) == 4
    assert calls[0]["workers"] == 9
    assert calls[0]["reader_refresh"] is False
    assert len(calls[0]["doc_ps"]) == 3
    assert all(call["workers"] == 1 for call in calls[1:])
    assert all(call["reader_refresh"] is True for call in calls[1:])
    clean_doc_ps = [call["doc_ps"][0].lower() for call in calls[1:]]
    assert clean_doc_ps == ["<quantity_1> daily", "a biguanide daily",
                            "a medication daily"]
    assert sum("biguanide" in doc_p for doc_p in clean_doc_ps) == 1


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


def test_train_roundtrip_threads_optional_carrier_fields(monkeypatch):
    doc = _with_carrier(_doc())
    captured = []

    def fake_roundtrip_carrier(jobs, workers=1):
        captured.extend(jobs)
        return [{"recall": float(i % 2), "out_p": "", "out_final": "", "f1s": []}
                for i, _ in enumerate(jobs)]

    monkeypatch.setattr(tr, "roundtrip_batch", fake_roundtrip_carrier)

    tr.train_roundtrip([doc], tr.RankerPolicy(), G=2, epochs=1, lr=0.01,
                       entropy_coef=0.0, kl_coef=0.0, ref=None,
                       rt_workers=1, seed=0)

    assert len(captured) == 2
    assert all(j["ladder"] == doc["ladder"] for j in captured)
    assert all(j["decisions"] == doc["decisions"] for j in captured)
    assert all(j["out_hi"] == doc["out_hi"] for j in captured)
    assert all(j["schema"] is True for j in captured)


def test_counterfactual_terms_threads_optional_carrier_fields(monkeypatch):
    doc = _with_carrier(_doc())
    choice = {"metformin": doc["spans"][0]["actions"][0]}
    logps = [torch.tensor(0.0, requires_grad=True)]
    captured = []

    def fake_roundtrip_carrier(jobs, workers=1):
        captured.extend(jobs)
        return [{"recall": 0.0, "out_p": "", "out_final": "", "f1s": []} for _ in jobs]

    monkeypatch.setattr(tr, "roundtrip_batch", fake_roundtrip_carrier)

    term, n_cf = tr.counterfactual_terms(doc, tr.RankerPolicy(), choice, logps, 1.0,
                                         frac=1.0, rng=__import__("random").Random(0),
                                         rt_workers=1)

    assert n_cf == 1
    assert isinstance(term, torch.Tensor)
    assert captured[0]["ladder"] == doc["ladder"]
    assert captured[0]["decisions"] == doc["decisions"]
    assert captured[0]["out_hi"] == doc["out_hi"]
    assert captured[0]["schema"] is True


def test_counterfactual_terms_excludes_span_free_decisions(monkeypatch):
    doc = _with_carrier(_doc())
    doc["decisions"] = [
        {"q": "Which route?", "options": ["endocrinology", "primary care"],
         "gold": "primary care", "span_ids": ["metformin"]},
        {"q": "Which billing path?", "options": ["routine", "complex"],
         "gold": "routine", "span_ids": []},
    ]
    choice = {"metformin": doc["spans"][0]["actions"][0]}
    logps = [torch.tensor(-0.7, requires_grad=True)]
    captured = []

    def fake_roundtrip_carrier(jobs, workers=1):
        captured.extend(jobs)
        out = []
        for job in jobs:
            has_span_free = any(not d.get("span_ids") for d in job.get("decisions", []))
            span_placeholdered = "<QUANTITY_1>" in job["doc_p"]
            reward = 0.0 if has_span_free and span_placeholdered else 1.0
            out.append({"recall": reward, "out_p": "", "out_final": "", "f1s": []})
        return out

    monkeypatch.setattr(tr, "roundtrip_batch", fake_roundtrip_carrier)

    term, n_cf = tr.counterfactual_terms(doc, tr.RankerPolicy(), choice, logps, base_r=999.0,
                                         frac=1.0, rng=__import__("random").Random(0),
                                         rt_workers=1)

    assert n_cf == 1
    assert term.item() == pytest.approx(0.0)
    assert len(captured) == 2
    assert all(job["decisions"] == [doc["decisions"][0]] for job in captured)
