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
