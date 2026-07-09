import importlib.util
import json
import sys
from pathlib import Path

from cloak.detect import Span


def _load_gate_module():
    repo = Path(__file__).resolve().parents[3]
    script = repo / "scripts" / "latticecloak_detection_gate.py"
    spec = importlib.util.spec_from_file_location("latticecloak_detection_gate_under_test", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _tab_doc(doc_id):
    text = "Alice met Paris."
    return {
        "id": doc_id,
        "text": text,
        "annotations": {
            "ann": {
                "entity_mentions": [
                    {
                        "start_offset": 0,
                        "end_offset": 5,
                        "identifier_type": "DIRECT",
                        "entity_type": "PERSON",
                        "entity_id": f"{doc_id}-alice",
                    },
                    {
                        "start_offset": 10,
                        "end_offset": 15,
                        "identifier_type": "QUASI",
                        "entity_type": "LOC",
                        "entity_id": f"{doc_id}-paris",
                    },
                ]
            }
        },
    }


def test_checkpoint_threshold_sweep_reuses_threshold_invariant_detector_work(tmp_path, monkeypatch):
    """Regression for docs/issues/performance.md: thresholds must be in-memory filters.

    The expensive detector pass should run once per checkpoint over the corpus, not once per
    checkpoint/threshold cell. This uses a fake detector so the test checks sweep structure rather
    than wall-clock timing or GPU availability.
    """
    gate = _load_gate_module()
    corpus = tmp_path / "tab_dev.json"
    corpus.write_text(json.dumps([_tab_doc("d1"), _tab_doc("d2")]))

    instances = []

    class CountingDetector:
        def __init__(self, gliner_model, threshold=0.3, fine_dem=False):
            self.gliner_model = gliner_model
            self.threshold = threshold
            self.fine_dem = fine_dem
            self.calls = []
            instances.append(self)

        def detect(self, text):
            self.calls.append(text)
            return [
                Span(0, 5, "Alice", "PERSON", 0.95, "gliner"),
                Span(10, 15, "Paris", "LOC", 0.90, "gliner"),
            ]

    monkeypatch.setattr(gate, "Detector", CountingDetector)
    checkpoints = ["checkpoint-a", "checkpoint-b"]
    thresholds = [0.02, 0.30]
    for checkpoint in checkpoints:
        for threshold in thresholds:
            monkeypatch.setattr(
                sys,
                "argv",
                [
                    "latticecloak_detection_gate.py",
                    "--corpus",
                    str(corpus),
                    "--gliner-model",
                    checkpoint,
                    "--threshold",
                    str(threshold),
                    "--out",
                    str(tmp_path / f"{checkpoint}-{threshold}.json"),
                ],
            )
            gate.main()

    total_detect_calls = sum(len(det.calls) for det in instances)
    assert total_detect_calls == 4, (
        "a 2-checkpoint x 2-threshold x 2-doc sweep should run detector work "
        "only 2 checkpoints x 2 docs = 4 times; threshold cells must reuse cached spans"
    )
