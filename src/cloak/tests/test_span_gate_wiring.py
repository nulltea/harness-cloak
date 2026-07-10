import json
import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

from calibrate_span_gate import choose_points, margin_scores


def test_margin_scores_and_choose_points_bars():
    # keeps at pos .9 / neg .1 ; drops at pos .1 / neg .9 -> separable
    keeps = [(0.9, 0.1)] * 100
    drops = [(0.1, 0.9)] * 50
    sweep, points = choose_points(keeps, drops,
                                  floors=[0.5], margins=[0.2],
                                  production_false_drop=0.001, miner_precision=0.99)
    assert points["production"] == {"floor": 0.5, "margin": 0.2}
    assert points["miner"] == {"floor": 0.5, "margin": 0.2}
    row = sweep[0]
    assert row["false_drop_rate"] == 0.0 and row["drop_recall"] == 1.0
    # inseparable data -> no production point
    _, points2 = choose_points([(0.1, 0.9)] * 100, [(0.1, 0.9)] * 100,
                               floors=[0.5], margins=[0.2],
                               production_false_drop=0.001, miner_precision=0.99)
    assert "production" not in points2


def test_margin_scores_pos_neg_maxima():
    # two keep surfaces embedded to fixed vectors; pos = max cos to index, neg = max cos to neg.
    vecs = {"a": np.array([1.0, 0.0], dtype=np.float32),
            "b": np.array([0.0, 1.0], dtype=np.float32)}
    index = np.array([[1.0, 0.0]], dtype=np.float32)   # aligns with "a"
    neg = np.array([[0.0, 1.0]], dtype=np.float32)      # aligns with "b"
    scores = margin_scores(["a", "b"], index, neg, lambda ts: np.array([vecs[t] for t in ts]))
    assert scores[0][0] == 1.0 and scores[0][1] == 0.0   # "a": pos high, neg low -> keep
    assert scores[1][0] == 0.0 and scores[1][1] == 1.0   # "b": pos low, neg high -> drop


def test_miner_gate_wiring_drop_retype_keep(monkeypatch, tmp_path):
    import build_mined_lattice_profiles as m
    from cloak.span_gate import GateDecision
    from cloak.profile_match import span_key

    decisions = {
        span_key("junky fragment", "injury"): GateDecision("drop", "margin"),
        span_key("flurbectomy", "injury"): GateDecision("retype", "retype",
                                                        new_type="medical-procedure"),
        span_key("blorbitis", "injury"): GateDecision("keep", "open"),
    }
    monkeypatch.setattr(m, "gate_spans", lambda items, point, **kw: decisions)
    spans = [m.DetectedSpan(s, "injury", "doc1", 0.9)
             for s in ("junky fragment", "flurbectomy", "blorbitis")]
    rows, stats = m.build_rows_for_test(spans)   # thin seam added in Step 3
    assert stats["gate_dropped"] == 1 and stats["gate_retyped"] == 1
    assert ("medical-procedure" in rows) and ("blorbitis" in rows.get("injury", {}))
    assert "gate_fingerprint" in stats


def test_runtime_gate_drop_retype_keep(monkeypatch):
    from cloak import detect
    from cloak.span_gate import GateDecision
    from cloak.profile_match import span_key

    decisions = {
        span_key("junky fragment", "injury"): GateDecision("drop", "margin"),
        span_key("flurbectomy", "injury"): GateDecision("retype", "retype",
                                                        new_type="medical-procedure"),
        span_key("blorbitis", "injury"): GateDecision("keep", "open"),
    }
    import cloak.span_gate as sg
    monkeypatch.setattr(sg, "gate_spans", lambda items, point, **kw: decisions)
    spans = [
        detect.Span(0, 3, "junky fragment", "injury", 0.9, "gliner"),
        detect.Span(4, 7, "flurbectomy", "injury", 0.8, "gliner"),
        detect.Span(8, 11, "blorbitis", "injury", 0.7, "gliner"),
        detect.Span(12, 15, "springtown", "GPE", 0.6, "gliner"),   # bypasses gate
    ]
    out = detect._apply_negative_filter(spans)
    by_text = {s.text: s for s in out}
    assert "junky fragment" not in by_text            # dropped
    assert by_text["flurbectomy"].type == "medical-procedure"  # retyped
    assert by_text["flurbectomy"].score == 0.8 and by_text["flurbectomy"].source == "gliner"
    assert by_text["blorbitis"].type == "injury"      # kept
    assert by_text["springtown"].type == "GPE"        # bypassed unchanged
