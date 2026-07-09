import importlib.util
from pathlib import Path


def _load_cli():
    path = Path("scripts/run_lattice_producer.py")
    spec = importlib.util.spec_from_file_location("run_lattice_producer", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_lattice_producer_cli_accepts_repeated_and_comma_separated_categories():
    cli = _load_cli()

    args = cli.parse_args(
        [
            "--run-dir",
            "results/lattice-producer/domain",
            "--profiles",
            "data/lattice_profiles/lattice_profiles.json",
            "--out",
            "data/lattice_profiles/proposed/domain.proposed.json",
            "--category",
            "drug,health-condition",
            "--category",
            "medical-procedure",
            "--thinking-budget-tokens",
            "1024",
        ]
    )

    assert args.category == ["drug", "health-condition", "medical-procedure"]
    assert args.thinking_budget_tokens == 1024


def test_lattice_producer_cli_defaults_to_bounded_thinking_budget():
    cli = _load_cli()

    args = cli.parse_args(
        [
            "--run-dir",
            "results/lattice-producer/domain",
            "--profiles",
            "data/lattice_profiles/lattice_profiles.json",
            "--out",
            "data/lattice_profiles/proposed/domain.proposed.json",
        ]
    )

    assert args.thinking_budget_tokens == 2048
