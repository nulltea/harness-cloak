import importlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def _module():
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        return importlib.import_module("build_arms_artifact")
    finally:
        sys.path.pop(0)


def test_build_arms_accepts_fine_detector_args(monkeypatch):
    mod = _module()
    args = mod.parse_args([
        "--detector-model", "data/models/pii_gliner_finedem/final",
        "--fine-dem",
        "--threshold", "0.22",
    ])

    seen = {}

    class FakeDetector:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    monkeypatch.setattr(mod, "Detector", FakeDetector)
    mod.make_detector(args)

    assert seen == {
        "gliner_model": "data/models/pii_gliner_finedem/final",
        "threshold": 0.22,
        "fine_dem": True,
    }
