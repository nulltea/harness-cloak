import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from build_probes import validate_probes  # noqa: E402


def test_validate_probes_keep_and_reject():
    cands = [{"surface": "metformin 500mg", "question": "What dose?"},   # keep
             {"surface": "March 3", "question": "When?"},                # ceiling reject
             {"surface": "chest pain", "question": "What symptom?"}]     # floor reject
    hi = [1.0, 0.2, 1.0]   # f1 vs ceiling anchor out_final(doc_orig)
    lo = [0.0, 0.0, 0.9]   # f1 vs floor anchor out_final(all_placeholder)
    kept, rej_c, rej_f = validate_probes(cands, hi, lo, th=0.5)
    assert [p["surface"] for p in kept] == ["metformin 500mg"]
    assert [p["surface"] for p in rej_c] == ["March 3"]
    assert [p["surface"] for p in rej_f] == ["chest pain"]
