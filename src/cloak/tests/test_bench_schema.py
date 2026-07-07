from pathlib import Path

from bench.schema import (
    BenchmarkConfig,
    BenchmarkItem,
    BenchmarkScores,
    BenchmarkTrace,
    PrivacyTarget,
    SensitiveSpan,
    StageOutput,
    jsonl_read,
    jsonl_write,
    stable_hash,
)


def _item() -> BenchmarkItem:
    return BenchmarkItem(
        item_id="clinical/mts/000001",
        domain="clinical",
        task="visit_note_generation",
        corpus="mts",
        doc_orig="Martha is a 50-year-old patient.",
        task_prompt_template="clinical",
        reference_outputs=["Martha is a 50-year-old patient."],
        gold_sensitive_spans=[
            SensitiveSpan(
                span_id="s1",
                surface="Martha",
                start=0,
                end=6,
                type="PERSON",
                identifier_class="DIRECT",
                subject_id="patient",
                task_relevance="gold_restated",
                reference_evidence=["Martha is a 50-year-old patient."],
            )
        ],
        privacy_targets=[
            PrivacyTarget(
                target_id="patient",
                known_to_attacker="document_context_only",
                secret_attributes=["name", "age"],
            )
        ],
    )


def test_item_round_trips_through_json():
    item = _item()

    got = BenchmarkItem.from_json(item.to_json())

    assert got == item
    assert got.gold_sensitive_spans[0].surface == "Martha"
    assert got.privacy_targets[0].secret_attributes == ["name", "age"]


def test_trace_round_trips_through_json():
    item = _item()
    stage = StageOutput(
        detected_spans=[{"surface": "Martha", "type": "PERSON", "start": 0, "end": 6}],
        R=[{"surface": "Martha", "replacement": "<PERSON_1>", "type": "PERSON"}],
        doc_p="<PERSON_1> is a 50-year-old patient.",
        out_p="<PERSON_1> is a patient.",
        out_final="Martha is a patient.",
        extractor_trace={"ph_swapped": 1},
    )
    trace = BenchmarkTrace(item=item, config_hash="abc123", stage=stage, metrics={"utility": 1.0})

    got = BenchmarkTrace.from_json(trace.to_json())

    assert got == trace
    assert got.stage.extractor_trace == {"ph_swapped": 1}


def test_config_hash_changes_when_remote_model_changes():
    base = BenchmarkConfig(
        suite="primary_utility",
        limit=2,
        seed=0,
        detector_version="current",
        substitutor_version="current",
        privacy_setting="tau=0.02",
        remote_model="gemma 4 (E4B)",
        extractor_version="current",
        attacker_version="offline-v1",
        output_dir="results/roundtrip_benchmark/test",
    )

    changed = base.replace(remote_model="stub")

    assert base.config_hash() != changed.config_hash()


def test_config_hash_and_json_include_all_model_arguments():
    base = BenchmarkConfig(
        suite="primary_utility",
        limit=2,
        seed=0,
        detector_version="current",
        substitutor_version="current",
        privacy_setting="tau=0.02",
        remote_model="gemma 4 (E4B)",
        extractor_version="current",
        attacker_version="offline-v1",
        output_dir="results/roundtrip_benchmark/test",
        detector_model="data/models/pii_gliner_finedem/final",
        extractor_model="all-MiniLM-L6-v2",
        attack_docp_model="offline-exact",
        attack_reconstruction_model="offline-exact",
        attack_leak_model="offline-exact",
    )

    got = BenchmarkConfig.from_json(base.to_json())
    changed = base.replace(detector_model="data/models/other-detector")

    assert got == base
    assert got.detector_model == "data/models/pii_gliner_finedem/final"
    assert got.attack_reconstruction_model == "offline-exact"
    assert base.config_hash() != changed.config_hash()


def test_stable_hash_is_order_invariant_for_dict_keys():
    assert stable_hash({"b": 2, "a": 1}) == stable_hash({"a": 1, "b": 2})
    assert stable_hash({"a": 1}) != stable_hash({"a": 2})


def test_jsonl_helpers(tmp_path: Path):
    path = tmp_path / "rows.jsonl"

    jsonl_write(path, [{"a": 1}, {"b": 2}])

    assert jsonl_read(path) == [{"a": 1}, {"b": 2}]


def test_scores_round_trip_keeps_frontier_rows():
    scores = BenchmarkScores(
        config_hash="abc",
        stage_metrics=[{"domain": "clinical", "rougeL": 0.7}],
        utility_metrics={"mean_rougeL": 0.7},
        privacy_metrics={"realized_privacy": 0.9},
        frontier=[{"method": "all_placeholder", "utility": 0.4, "realized_privacy": 0.9}],
        gates=[{"name": "unsupported_insertions", "passed": True}],
    )

    assert BenchmarkScores.from_json(scores.to_json()) == scores
