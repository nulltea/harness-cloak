import json
import subprocess
import sys


def test_cli_dry_run_writes_manifest_and_items(tmp_path):
    out = tmp_path / "bench"
    cmd = [
        sys.executable,
        "scripts/run_roundtrip_benchmark.py",
        "--suite",
        "primary_utility",
        "--limit",
        "2",
        "--output-dir",
        str(out),
        "--dry-run",
    ]

    res = subprocess.run(cmd, check=True, text=True, capture_output=True, env={"PYTHONPATH": "src"})

    manifest = out / "manifest.json"
    items = out / "items.jsonl"
    assert manifest.exists()
    assert items.exists()
    payload = json.loads(manifest.read_text())
    assert payload["suite"] == "primary_utility"
    assert "dry-run: wrote 2 items" in res.stdout


def test_cli_records_all_model_arguments_in_manifest(tmp_path):
    out = tmp_path / "bench"
    cmd = [
        sys.executable,
        "scripts/run_roundtrip_benchmark.py",
        "--suite",
        "primary_utility",
        "--limit",
        "1",
        "--remote-model",
        "stub",
        "--detector-model",
        "data/models/pii_gliner_finedem/final",
        "--extractor-model",
        "all-MiniLM-L6-v2",
        "--attack-docp-model",
        "offline-docp",
        "--attack-reconstruction-model",
        "offline-reconstruct",
        "--attack-leak-model",
        "offline-leak",
        "--output-dir",
        str(out),
        "--dry-run",
    ]

    subprocess.run(cmd, check=True, text=True, capture_output=True, env={"PYTHONPATH": "src"})

    payload = json.loads((out / "manifest.json").read_text())
    assert payload["detector_model"] == "data/models/pii_gliner_finedem/final"
    assert payload["extractor_model"] == "all-MiniLM-L6-v2"
    assert payload["attack_docp_model"] == "offline-docp"
    assert payload["attack_reconstruction_model"] == "offline-reconstruct"
    assert payload["attack_leak_model"] == "offline-leak"


def test_cli_live_current_detector_requires_detector_model(tmp_path):
    out = tmp_path / "bench"
    cmd = [
        sys.executable,
        "scripts/run_roundtrip_benchmark.py",
        "--suite",
        "primary_utility",
        "--limit",
        "1",
        "--detector-version",
        "current",
        "--substitutor",
        "all_placeholder",
        "--remote-model",
        "stub",
        "--stub-remote",
        "--output-dir",
        str(out),
    ]

    res = subprocess.run(cmd, text=True, capture_output=True, env={"PYTHONPATH": "src"})

    assert res.returncode != 0
    assert "--detector-model is required" in res.stderr


def test_cli_stub_remote_writes_traces(tmp_path):
    out = tmp_path / "bench"
    cmd = [
        sys.executable,
        "scripts/run_roundtrip_benchmark.py",
        "--suite",
        "primary_utility",
        "--limit",
        "2",
        "--detector-version",
        "gold",
        "--substitutor",
        "all_placeholder",
        "--remote-model",
        "stub",
        "--stub-remote",
        "--output-dir",
        str(out),
    ]

    res = subprocess.run(cmd, check=True, text=True, capture_output=True, env={"PYTHONPATH": "src"})

    assert (out / "traces.jsonl").exists()
    assert (out / "stage_metrics.json").exists()
    assert (out / "privacy_metrics.json").exists()
    assert (out / "utility_metrics.json").exists()
    assert (out / "matched_privacy_frontier.json").exists()
    assert (out / "report.md").exists()
    assert "scored 2 traces" in res.stdout
