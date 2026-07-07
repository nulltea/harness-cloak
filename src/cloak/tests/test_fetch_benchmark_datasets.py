import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "download"))

from fetch_benchmark_datasets import (
    FileSpec,
    SOURCES,
    apply_includes,
    enforce_size_budget,
    hf_file_specs,
    known_size,
    mendeley_file_specs,
    selected_datasets,
    source_report,
    write_manifest,
)


def test_source_report_lists_requested_public_datasets():
    report = source_report()

    assert report["sources"]["primock57"]["license"] == "CC BY 4.0"
    assert report["sources"]["rat-bench"]["license"] == "MIT"
    assert report["sources"]["synthetic-financial-pii"]["license"] == "CC BY 4.0"
    assert report["sources"]["pii-bench"]["license"] == "Apache-2.0"
    assert report["sources"]["pii-bench"]["repo_id"] == "Pritesh-2711/pii-bench"


def test_hf_file_specs_use_resolve_urls_and_skip_directories():
    rows = [
        {"type": "directory", "path": "data", "size": 0},
        {"type": "file", "path": "README.md", "size": 10},
        {"type": "file", "path": "data/test file.jsonl", "size": 20},
    ]

    specs = hf_file_specs("owner/repo", "main", rows)

    assert [s.path for s in specs] == ["README.md", "data/test file.jsonl"]
    assert specs[0].url == "https://huggingface.co/datasets/owner/repo/resolve/main/README.md"
    assert specs[1].url.endswith("/data/test%20file.jsonl")
    assert known_size(specs) == 30


def test_mendeley_file_specs_preserve_download_url_and_checksum():
    rows = [
        {
            "filename": "Training_Set.xlsx",
            "size": 123,
            "content_details": {
                "download_url": "https://data.mendeley.com/public-files/file_downloaded",
                "sha256_hash": "abc123",
            },
        },
        {"filename": "missing-url.xlsx", "content_details": {}},
    ]

    specs = mendeley_file_specs(rows)

    assert specs == [
        FileSpec(
            path="Training_Set.xlsx",
            url="https://data.mendeley.com/public-files/file_downloaded",
            size=123,
            sha256="abc123",
        )
    ]


def test_include_globs_and_size_budget_guard():
    files = [
        FileSpec("README.md", "https://example.test/readme", 10),
        FileSpec("data/train.jsonl", "https://example.test/train", 100),
        FileSpec("data/test.jsonl", "https://example.test/test", 20),
    ]

    selected = apply_includes(files, ["data/test*"])

    assert selected == [files[2]]
    enforce_size_budget("demo", selected, max_bytes=20)
    try:
        enforce_size_budget("demo", files, max_bytes=20)
    except SystemExit as exc:
        assert "above --max-bytes" in str(exc)
    else:
        raise AssertionError("expected over-budget selection to fail")


def test_selected_datasets_deduplicates_all_and_explicit():
    selected = selected_datasets(["pii-bench", "pii-bench"], all_sources=True)

    assert selected[0] == "pii-bench"
    assert set(SOURCES).issubset(selected)
    assert len(selected) == len(set(selected))


def test_write_manifest_records_license_source_and_files(tmp_path):
    src = SOURCES["rat-bench"]
    files = [FileSpec("README.md", "https://example.test/readme", 10)]
    results = [{"path": "README.md", "size": 10, "status": "downloaded"}]

    path = write_manifest(
        tmp_path,
        src,
        {"revision": "abc"},
        files,
        results,
        dry_run=False,
    )

    manifest = json.loads(path.read_text())
    assert manifest["dataset"] == "rat-bench"
    assert manifest["license"] == "MIT"
    assert manifest["source"]["revision"] == "abc"
    assert manifest["selected_files"][0]["path"] == "README.md"
