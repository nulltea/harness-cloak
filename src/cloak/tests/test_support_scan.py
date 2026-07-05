import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "spikes"))
from roundtrip_support_scan import scan_verdict  # noqa: E402


def _row(delta, up=0, down=0):
    return {"doc_id": "d", "surface": "s", "from": 0, "to": 1, "delta": delta,
            "probe_flips_up": up, "probe_flips_down": down}


def test_pass_needs_both_directions_and_magnitude():
    rows = [_row(0.15, up=1), _row(-0.2, down=1), _row(0.0)]
    v = scan_verdict(rows, mean_probes=10.0)
    assert v["verdict"] == "PASS" and v["n_up"] == 1 and v["n_down"] == 1


def test_fail_one_direction_only():
    rows = [_row(-0.2, down=1), _row(-0.1, down=1)]
    assert scan_verdict(rows, mean_probes=10.0)["verdict"] == "FAIL"


def test_fail_below_quantization():
    rows = [_row(0.01, up=0), _row(-0.01, down=0)]   # |delta| < 1/mean_probes = 0.1
    assert scan_verdict(rows, mean_probes=10.0)["verdict"] == "FAIL"
