import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))


def test_miner_gate_wiring_drop_retype_keep(monkeypatch, tmp_path):
    import build_mined_lattice_profiles as m
    from cloak.detection.span_gate import GateDecision
    from cloak.lattice.profile_match import span_key

    decisions = {
        span_key("junky fragment", "injury"): GateDecision("drop", "nli"),
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


def test_miner_gate_retype_reapplies_type_handling(monkeypatch):
    import build_mined_lattice_profiles as m
    from cloak.detection.span_gate import GateDecision
    from cloak.lattice.profile_match import span_key

    decisions = {
        # retyped into "drug": surface must now be dose-stripped
        span_key("wingivax 20 mg", "injury"): GateDecision("retype", "retype", new_type="drug"),
        # retyped into a type where the surface is generic: must be skipped, not added
        span_key("procedure", "injury"): GateDecision("retype", "retype",
                                                      new_type="medical-procedure"),
    }
    monkeypatch.setattr(m, "gate_spans", lambda items, point, **kw: decisions)
    spans = [m.DetectedSpan("wingivax 20 mg", "injury", "doc1", 0.9),
             m.DetectedSpan("procedure", "injury", "doc1", 0.9)]
    rows, stats = m.build_rows_for_test(spans)
    assert "wingivax" in rows.get("drug", {})           # dose-stripped for the NEW type
    assert "wingivax 20 mg" not in rows.get("drug", {})
    assert "medical-procedure" not in rows              # generic-for-new-type -> skipped
    assert stats["generic_skipped"] == 1 and stats["gate_retyped"] == 2


def test_runtime_gate_drop_retype_keep(monkeypatch):
    from cloak.detection import detect
    from cloak.detection.span_gate import GateDecision
    from cloak.lattice.profile_match import span_key

    decisions = {
        span_key("junky fragment", "injury"): GateDecision("drop", "nli"),
        span_key("flurbectomy", "injury"): GateDecision("retype", "retype",
                                                        new_type="medical-procedure"),
        span_key("blorbitis", "injury"): GateDecision("keep", "open"),
    }
    import cloak.detection.span_gate as sg
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
