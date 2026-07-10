import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

from calibrate_entity_merge_gate import build_eval_pairs, choose_threshold
from cloak.lattice_producer.reference_sources import DoidNode


def _nodes():
    return {
        "DOID:1": DoidNode("DOID:1", "blorbitis", parents=["DOID:9"],
                           exact_synonyms=["blorb inflammation"]),
        "DOID:2": DoidNode("DOID:2", "glimmerosis", parents=["DOID:9"]),
        "DOID:3": DoidNode("DOID:3", "flurbosis", parents=["DOID:9"]),
        "DOID:8": DoidNode("DOID:8", "old thing", parents=["DOID:9"], obsolete=True),
        "DOID:9": DoidNode("DOID:9", "organ disease"),
    }


def test_build_eval_pairs_positives_are_synonyms_negatives_are_siblings():
    pos, neg = build_eval_pairs(_nodes(), sample=100, seed=0)
    assert ("blorbitis", "blorb inflammation") in pos
    assert all(a != b for a, b in neg)
    sib_names = {frozenset(p) for p in neg}
    assert frozenset(("glimmerosis", "flurbosis")) in sib_names
    assert not any("old thing" in p for p in [*pos, *neg])   # obsolete excluded


def test_choose_threshold_requires_precision_bar_and_recall_floor():
    scored = [(0.9, True), (0.8, True), (0.7, False), (0.6, True), (0.2, False)]
    # at t=0.8: P=1.0, R=2/3 -> chosen; lower t admits the 0.7 negative
    assert choose_threshold(scored, precision_bar=0.999, recall_floor=0.10) == 0.8
    # unreachable bar -> None (gate ships disabled)
    assert choose_threshold([(0.9, False), (0.8, True)], 0.999, 0.10) is None


def test_dedupe_cli_rewrites_artifact_and_report(tmp_path):
    import dedupe_lattice_profile_entries as cli
    artifact = {"schema_version": 1, "created": "2026-07-10", "sources": {}, "profiles": {
        "health-condition": {
            "blorbitis": {"aliases": [], "levels": ["organ disease"],
                          "source_ids": ["t:1"], "count": 10.0},
            "blorb inflammation": {"aliases": [], "levels": ["organ disease"],
                                   "source_ids": ["t:2"], "count": 4.0}}}}
    profiles = tmp_path / "profiles.json"
    profiles.write_text(json.dumps(artifact))
    obo = tmp_path / "mini.obo"
    obo.write_text('[Term]\nid: DOID:1\nname: blorbitis\n'
                   'synonym: "blorb inflammation" EXACT []\n')
    report_out = tmp_path / "report.json"
    cli.main(["--profiles", str(profiles), "--obo", f"health-condition={obo}",
              "--report-out", str(report_out), "--skip-embindex", "--no-embed-blocking"])
    got = json.loads(profiles.read_text())
    assert set(got["profiles"]["health-condition"]) == {"blorbitis"}
    report = json.loads(report_out.read_text())
    assert report["types"]["health-condition"]["merged"]
