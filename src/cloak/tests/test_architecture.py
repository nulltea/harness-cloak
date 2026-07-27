"""Import contracts for the package structure (refactor plan Phase 0/4).

Contracts are added as the refactor phases land; each is a static AST scan so
violations fail fast without importing heavy modules.
"""
import ast
from pathlib import Path

CLOAK_ROOT = Path(__file__).resolve().parents[1]

# The stage packages the regroup (Phase 4) established.
STAGE_PACKAGES = ("detection", "lattice", "qa", "reward", "ranker")


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


def _package_modules(package: str):
    for path in (CLOAK_ROOT / package).rglob("*.py"):
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

    `qa/scoring` owns the reader pin and `UTILITY_SCORER_VERSION`, both of which sit
    inside every utility-cache identity; importing the compiler, the teacher transport,
    or the freezer would let a build-time edit move a cache key.
    """
    forbidden = {"cloak.qa.builder", "cloak.qa.teacher", "cloak.qa.freeze",
                 "cloak.qa.review"}
    leaked = forbidden & _module_imports(CLOAK_ROOT / "qa" / "scoring.py")
    assert not leaked, f"qa/scoring must not import {sorted(leaked)}"


def test_reward_never_imports_the_ranker_policy():
    """`cloak.reward` scores rollouts; it must not depend on who produced them.

    The one allowed edge is `cloak.ranker.environment` — the frozen MDP data contract
    (documents/decisions/actions) plus its renderer, which the reward path needs to turn
    an action vector into `doc_p`. Everything else under `cloak.ranker` is policy,
    training, or diagnostics: importing it would make the reward pin depend on the
    learner and invert the dependency direction.
    """
    allowed = {"cloak.ranker.environment"}
    violations = {}
    for path in _package_modules("reward"):
        leaked = sorted(
            name for name in _module_imports(path)
            if name.startswith("cloak.ranker") and name not in allowed
        )
        if leaked:
            violations[str(path.relative_to(CLOAK_ROOT.parent))] = leaked
    assert not violations, f"cloak/reward imports ranker internals: {violations}"


def test_ranker_environment_stays_policy_free():
    """The shared environment/renderer must not reach back into the policy stack.

    Without this, the `reward -> ranker.environment` edge above would be a back door to
    the whole learner (and would drag torch into the reward path).
    """
    leaked = sorted(
        name for name in _module_imports(CLOAK_ROOT / "ranker" / "environment.py")
        if name.split(".")[0] in {"cloak", "torch"}
        and not name.startswith(("cloak.corpora", "cloak.runtime_types"))
    )
    assert not leaked, f"ranker/environment must stay policy-free, imports {leaked}"


def test_the_retired_train_package_is_gone():
    assert not (CLOAK_ROOT / "train").exists(), "cloak/train came back"


def test_stage_packages_own_every_non_core_module():
    """Only the shared core stays at the `cloak` root; stages own the rest."""
    core = {"__init__.py", "llm.py", "concurrent.py", "runtime_types.py",
            "corpora.py", "tasks.py"}
    stray = sorted(
        path.name for path in CLOAK_ROOT.glob("*.py") if path.name not in core
    )
    assert not stray, f"modules loose at the cloak root: {stray}"
    missing = [name for name in STAGE_PACKAGES if not (CLOAK_ROOT / name).is_dir()]
    assert not missing, f"missing stage packages: {missing}"


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
