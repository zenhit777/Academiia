"""Tests du squelette d-OPD : paquet, configuration, règle de dépendance (J12).

Aucun test ne charge de modèle, n'utilise torch ou un GPU. Lancer :

    pytest modules/dopd/tests -q
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import academiia.dopd as dopd
from academiia.dopd import DOPDConfig

PACKAGE_DIR = Path(__file__).resolve().parents[1] / "dopd"
# ADR-001 : un module importe academiia.core, jamais un autre module du dépôt.
ALLOWED_REPO_IMPORTS = ("academiia.core",)
FORBIDDEN_HEAVY_IMPORTS = ("torch", "transformers", "deepspeed")


def _imported_modules(path: Path) -> set[str]:
    """Noms de modules importés statiquement par un fichier Python."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_package_importable_and_exports():
    assert dopd.__version__ == "0.1.0"
    for name in (
        "DOPDConfig",
        "TargetResult",
        "candidate_union",
        "compute_future_aware_target",
        "relative_future_deltas",
    ):
        assert name in dopd.__all__
        assert hasattr(dopd, name)


def test_package_never_imports_other_repo_modules():
    """ADR-001 : seul ``academiia.core`` est autorisé, tout autre module du
    dépôt est interdit — y compris les modules qui n'existent pas encore."""
    files = sorted(PACKAGE_DIR.glob("*.py"))
    assert files, "aucun fichier Python dans le paquet"
    violations = []
    for path in files:
        for module in _imported_modules(path):
            if not module.startswith("academiia."):
                continue
            if not any(module == allowed or module.startswith(allowed + ".")
                       for allowed in ALLOWED_REPO_IMPORTS):
                violations.append(f"{path.name}: {module}")
    assert not violations, f"import(s) interdit(s) : {violations}"


def test_package_never_manipulates_sys_path():
    """ADR-001 : aucun aménagement de ``sys.path`` dans le paquet."""
    violations = []
    for path in sorted(PACKAGE_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute):
                continue
            if node.attr not in {"insert", "append", "extend", "remove"}:
                continue
            target = node.value
            if isinstance(target, ast.Attribute) and target.attr == "path":
                violations.append(f"{path.name}: sys.path.{node.attr}")
    assert not violations, f"manipulation de sys.path : {violations}"


def test_package_never_imports_heavy_training_stack():
    """J12 reste CPU : le squelette ne doit pas attirer torch/transformers/deepspeed."""
    violations = []
    for path in sorted(PACKAGE_DIR.glob("*.py")):
        for module in _imported_modules(path):
            root = module.split(".", 1)[0]
            if root in FORBIDDEN_HEAVY_IMPORTS:
                violations.append(f"{path.name}: {module}")
    assert not violations, f"dépendance lourde importée : {violations}"


def test_config_defaults_match_paper():
    cfg = DOPDConfig()
    assert cfg.block_size == 4
    assert cfg.candidate_k == 16
    assert cfg.correctness_gate is True
    assert cfg.denoising_steps == 4
    assert cfg.max_response_length == 4_000
    assert cfg.scorer_dtype == "bfloat16"
    assert cfg.arithmetic_dtype == "float32"
    assert cfg.candidate_union_bound == 32
    assert cfg.max_scorer_branches == 33  # union (<= 2k) + reference


def test_config_arithmetic_dtype_is_the_target_default():
    """Le dtype de la config n'est pas descriptif : c'est le défaut de la cible."""
    import inspect

    from academiia.dopd import DEFAULT_ARITHMETIC_DTYPE, compute_future_aware_target

    assert DEFAULT_ARITHMETIC_DTYPE == DOPDConfig().arithmetic_dtype
    default = inspect.signature(compute_future_aware_target).parameters[
        "arithmetic_dtype"
    ].default
    assert default == DEFAULT_ARITHMETIC_DTYPE


def test_config_to_dict_is_serialisable():
    cfg = DOPDConfig(block_size=8, candidate_k=4)
    data = cfg.to_dict()
    assert data["block_size"] == 8
    assert data["candidate_k"] == 4
    assert data["correctness_gate"] is True
    # Le futur manifeste de lignee (ADR-002) pourra embarquer ce dict tel quel.
    assert set(data) == {
        "block_size", "candidate_k", "correctness_gate", "denoising_steps",
        "max_response_length", "scorer_dtype", "arithmetic_dtype",
    }


@pytest.mark.parametrize(
    "kwargs, exc",
    [
        ({"block_size": 0}, ValueError),
        ({"candidate_k": 0}, ValueError),
        ({"denoising_steps": 0}, ValueError),
        ({"max_response_length": 0}, ValueError),
        ({"block_size": 4.0}, TypeError),
        ({"correctness_gate": 1}, TypeError),
        ({"scorer_dtype": ""}, ValueError),
        ({"arithmetic_dtype": ""}, ValueError),
        ({"arithmetic_dtype": "float16"}, ValueError),
    ],
)
def test_config_rejects_invalid_values(kwargs, exc):
    with pytest.raises(exc):
        DOPDConfig(**kwargs)
