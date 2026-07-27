"""Import contracts for the package structure (refactor plan Phase 0).

Contracts are added as the refactor phases land; each is a static AST scan so
violations fail fast without importing heavy modules.
"""
import ast
from pathlib import Path

CLOAK_ROOT = Path(__file__).resolve().parents[1]


def _module_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
    return names


def _cloak_modules():
    for path in CLOAK_ROOT.rglob("*.py"):
        if "tests" in path.relative_to(CLOAK_ROOT).parts:
            continue
        yield path


def test_cloak_never_imports_inferdpt():
    violations = [
        str(path.relative_to(CLOAK_ROOT.parent))
        for path in _cloak_modules()
        if any(name.split(".")[0] == "inferdpt" for name in _module_imports(path))
    ]
    assert not violations, f"cloak modules import retired inferdpt: {violations}"


def test_qa_scoring_is_the_bottom_of_the_qa_stack():
    """The runtime slice the training loop executes must stay build-time-free.

    `qa_scoring` owns the reader pin and `UTILITY_SCORER_VERSION`, both of which sit
    inside every utility-cache identity; importing the compiler, the teacher transport,
    or the freezer would let a build-time edit move a cache key.
    """
    forbidden = {"cloak.train.qa_builder", "cloak.train.qa_teacher", "cloak.train.qa_freeze",
                 "cloak.train.qa_review"}
    leaked = forbidden & _module_imports(CLOAK_ROOT / "train" / "qa_scoring.py")
    assert not leaked, f"qa_scoring must not import {sorted(leaked)}"


def test_cloak_never_imports_scripts():
    violations = [
        str(path.relative_to(CLOAK_ROOT.parent))
        for path in _cloak_modules()
        if any(
            name.split(".")[0] in {"scripts", "train_interactive_ranker"}
            for name in _module_imports(path)
        )
    ]
    assert not violations, f"cloak modules import script namespaces: {violations}"
