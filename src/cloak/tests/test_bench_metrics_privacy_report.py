from bench.metrics import bootstrap_ci, echo_labels, restoration_metrics, score_traces, utility_metrics
from bench.privacy import (
    attribute_attacker,
    leak_through_attacker,
    matched_privacy_bins,
    realized_privacy_score,
    reconstruction_attacker,
)
from bench.report import acceptance_gates, write_markdown_report
from bench.schema import BenchmarkConfig, BenchmarkItem, BenchmarkScores, BenchmarkTrace, StageOutput


def _item() -> BenchmarkItem:
    return BenchmarkItem(
        item_id="bio/1",
        domain="biography",
        task="bio_summary",
        corpus="wikibio",
        doc_orig="Martha is a cardiologist in Oslo.",
        task_prompt_template="wikibio",
        reference_outputs=["Martha is a cardiologist."],
        gold_sensitive_spans=[],
        privacy_targets=[],
    )


def _trace(
    out_p="<PERSON_1> is a healthcare worker.",
    out_final="Martha is a cardiologist.",
    doc_p="<PERSON_1> is a healthcare worker in a city.",
) -> BenchmarkTrace:
    stage = StageOutput(
        detected_spans=[],
        R=[
            {"surface": "Martha", "replacement": "<PERSON_1>", "type": "PERSON", "action": "placeholder"},
            {"surface": "cardiologist", "replacement": "healthcare worker", "type": "profession", "action": "generalize"},
            {"surface": "Oslo", "replacement": "a city", "type": "LOC", "action": "generalize"},
        ],
        doc_p=doc_p,
        out_p=out_p,
        out_final=out_final,
    )
    return BenchmarkTrace(item=_item(), config_hash="abc", stage=stage, metrics={})


def _config() -> BenchmarkConfig:
    return BenchmarkConfig(
        suite="primary_utility",
        limit=1,
        seed=0,
        detector_version="gold",
        substitutor_version="all_placeholder",
        privacy_setting="tau=0.02",
        remote_model="stub",
        extractor_version="current",
        attacker_version="offline-v1",
        output_dir="results/roundtrip_benchmark/test",
    )


def test_echo_labels_distinguish_echoed_and_absent():
    labels = echo_labels(_trace())
    by_surface = {row["surface"]: row["echo"] for row in labels}

    assert by_surface["Martha"] == "exact"
    assert by_surface["cardiologist"] == "exact"
    assert by_surface["Oslo"] == "absent"


def test_restoration_metrics_count_supported_recovery_only():
    metrics = restoration_metrics(_trace())

    assert metrics["echoed_span_recovery"] == 1.0
    assert metrics["unsupported_insertion_count"] == 0


def test_utility_metrics_use_reference_restated_sensitive_facts():
    metrics = utility_metrics(_trace())

    assert 0.0 <= metrics["rougeL"] <= 1.0
    assert metrics["sensitive_fact_recall"] == 1.0


def test_bootstrap_ci_is_deterministic_and_ordered():
    lo, hi = bootstrap_ci([0.0, 0.5, 1.0], seed=123, samples=100)

    assert 0.0 <= lo <= hi <= 1.0
    assert (lo, hi) == bootstrap_ci([0.0, 0.5, 1.0], seed=123, samples=100)


def test_privacy_attackers_score_doc_p_and_out_final_separately():
    trace = _trace(doc_p="Martha is a healthcare worker in a city.", out_final="Martha is in Oslo.")

    assert attribute_attacker(trace)["doc_p_exact_leaks"] == 1
    assert reconstruction_attacker(trace)["replacement_echoes"] == 2
    assert leak_through_attacker(trace)["out_final_exact_leaks"] == 2


def test_realized_privacy_and_bins():
    assert realized_privacy_score([{"attack_success": 0.0}, {"attack_success": 0.5}]) == 0.75
    rows = [{"method": "a", "realized_privacy": 0.91}, {"method": "b", "realized_privacy": 0.93}]

    assert list(matched_privacy_bins(rows, width=0.05).values()) == [rows]


def test_score_traces_combines_utility_privacy_and_frontier():
    scores = score_traces([_trace()], _config())

    assert scores.config_hash == _config().config_hash()
    assert scores.utility_metrics["mean_rougeL"] >= 0.0
    assert 0.0 <= scores.privacy_metrics["realized_privacy"] <= 1.0
    assert scores.frontier[0]["method"] == "all_placeholder"


def test_report_writes_frontier_and_gates(tmp_path):
    scores = BenchmarkScores(
        config_hash="abc",
        stage_metrics=[{"domain": "clinical", "unsupported_insertion_count": 0, "rougeL": 0.7}],
        utility_metrics={"mean_rougeL": 0.7, "mean_sensitive_fact_recall": 0.6},
        privacy_metrics={"realized_privacy": 0.9, "doc_p_attack_success": 0.1},
        frontier=[{"method": "all_placeholder", "realized_privacy": 0.9, "utility": 0.4}],
        gates=[],
    )

    gates = acceptance_gates(scores)
    path = write_markdown_report(scores, tmp_path)
    text = path.read_text()

    assert all(gate["passed"] for gate in gates)
    assert "Privacy-Utility Frontier" in text
    assert "Acceptance Gates" in text
    assert "all_placeholder" in text
