import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

from lattice_sources.categorical import alias_rows
from lattice_sources.common import ProfileRow
from lattice_sources.obo import rows_from_obo
from lattice_sources.occupation import rows_from_isco_csv, rows_from_onet_titles
from build_lattice_profiles import coverage_report, merge_rows


def test_onet_titles_to_profession_rows(tmp_path):
    src = tmp_path / "Alternate Titles.txt"
    src.write_text(
        "O*NET-SOC Code\tTitle\tAlternate Title\tShort Title\tSource(s)\n"
        "27-3023.00\tNews Analysts, Reporters, and Journalists\tReporter\tN\tsample\n"
    )

    rows = rows_from_onet_titles(src)

    assert rows[0].runtime_type == "profession"
    assert rows[0].surface == "news analysts, reporters, and journalists"
    assert "reporter" in rows[0].aliases
    assert "media worker" in rows[0].levels


def test_isco_csv_to_profession_rows(tmp_path):
    src = tmp_path / "isco.csv"
    src.write_text(
        "code,title,major_group\n"
        "2211,Generalist medical practitioners,Professionals\n"
    )

    rows = rows_from_isco_csv(src)

    assert rows[0].surface == "generalist medical practitioners"
    assert rows[0].levels == ["professional worker"]


def test_obo_rows_use_synonyms_and_family_roots(tmp_path):
    src = tmp_path / "doid.obo"
    src.write_text(
        "[Term]\n"
        "id: DOID:9351\n"
        "name: diabetes mellitus\n"
        "synonym: \"diabetes\" EXACT []\n"
        "is_a: DOID:28 ! endocrine system disease\n"
        "\n"
        "[Term]\n"
        "id: DOID:28\n"
        "name: endocrine system disease\n"
    )

    rows = rows_from_obo(src, "health-condition", {"DOID:28": "endocrine condition"})

    row = next(r for r in rows if r.surface == "diabetes mellitus")
    assert "diabetes" in row.aliases
    assert row.levels == ["endocrine condition"]


def test_categorical_alias_rows_have_no_levels():
    rows = alias_rows("marital-status", {"married": ["wedded", "spouse"]})

    assert rows[0].runtime_type == "marital-status"
    assert rows[0].surface == "married"
    assert rows[0].aliases == ["wedded", "spouse"]
    assert rows[0].levels == []


def test_merge_rows_combines_aliases_levels_and_source_ids():
    artifact = merge_rows([
        ProfileRow("profession", "journalist", ["reporter"], ["media worker"], ["esco:1"], 1000.0),
        ProfileRow("profession", "journalist", ["correspondent"], ["media worker"], ["onet:2"], 1200.0),
    ])

    row = artifact["profiles"]["profession"]["journalist"]
    assert row["aliases"] == ["correspondent", "reporter"]
    assert row["levels"] == ["media worker"]
    assert row["source_ids"] == ["esco:1", "onet:2"]
    assert row["count"] == 1200.0


def test_coverage_report_marks_placeholder_only_types_separately():
    art = {
        "schema_version": 1,
        "created": "2026-07-07",
        "sources": {},
        "profiles": {"profession": {"journalist": {"levels": ["media worker"]}}},
    }

    report = coverage_report(art)

    assert report["profile_counts"]["profession"] == 1
    assert "gender" in report["placeholder_only_types"]
    assert "demographic-other" in report["placeholder_first_types"]


def test_build_lattice_profiles_cli_smoke(tmp_path):
    raw = tmp_path / "raw"
    (raw / "onet").mkdir(parents=True)
    (raw / "onet" / "Alternate Titles.txt").write_text(
        "O*NET-SOC Code\tTitle\tAlternate Title\tShort Title\tSource(s)\n"
        "27-3023.00\tNews Analysts, Reporters, and Journalists\tReporter\tN\tsample\n"
    )
    out = tmp_path / "profiles.json"
    cov = tmp_path / "coverage.json"

    subprocess.run(
        [
            ".venv/bin/python",
            "-u",
            "scripts/build_lattice_profiles.py",
            "--raw-dir",
            str(raw),
            "--out",
            str(out),
            "--coverage-out",
            str(cov),
        ],
        check=True,
        env={"PYTHONPATH": "src:scripts"},
    )

    art = json.loads(out.read_text())
    assert art["profiles"]["profession"]
    assert json.loads(cov.read_text())["profile_counts"]["profession"] >= 1
