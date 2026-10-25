"""Tests CPU du micro-benchmark de budget RTX 5090 (J14).

Le script mesuré est un harnais GPU : ces tests n'exécutent **aucune** mesure
GPU. Ils vérifient uniquement l'arithmétique pure dont dépend la note de
budget (taille du cache KV) et l'invariant de sûreté du script (le modèle est
gelé, la sortie reste dans le dépôt).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# Garde de collecte : sans torch, le module entier est ignore (et non une
# erreur d'import). Ce fichier n'utilise pas torch directement.
pytest.importorskip("torch", reason="torch absent : tests GPU ignores")

_HARNESS_PATH = Path(__file__).resolve().parents[1] / "harness" / "bench_budget_rtx5090.py"
_SPEC = importlib.util.spec_from_file_location("dopd_harness_budget", _HARNESS_PATH)
assert _SPEC is not None and _SPEC.loader is not None
harness = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = harness
_SPEC.loader.exec_module(harness)


class _FakeConfig:
    """Configuration Qwen3 miniature : 2 couches, 8 têtes KV, head_dim 16."""

    hidden_size = 64
    num_hidden_layers = 2
    num_key_value_heads = 8
    num_attention_heads = 8
    head_dim = 16


def test_kv_cache_matches_the_closed_formula():
    """Le cache KV doit valoir 2 (K et V) × couches × lot × longueur × tête × dim."""
    config = _FakeConfig()
    expected = 2 * 2 * 4 * 1024 * 8 * 16 * 2 / 1024**2
    assert harness.kv_cache_mib(config, batch_size=4, sequence_length=1024) == expected


def test_kv_cache_scales_linearly_with_batch_and_context():
    """Le cache doit doubler quand le lot double, et quand la longueur double."""
    config = _FakeConfig()
    base = harness.kv_cache_mib(config, batch_size=8, sequence_length=1024)
    assert harness.kv_cache_mib(config, batch_size=16, sequence_length=1024) == 2 * base
    assert harness.kv_cache_mib(config, batch_size=8, sequence_length=2048) == 2 * base


class _FakeParameter:
    """Paramètre factice qui mémorise son état ``requires_grad``."""

    def __init__(self) -> None:
        self.requires_grad = True

    def requires_grad_(self, value: bool) -> "_FakeParameter":
        self.requires_grad = value
        return self


class _FakeFrozenModel:
    """Modèle factice exposant ``eval()`` et ``parameters()``."""

    def __init__(self) -> None:
        self.parameters_list = [_FakeParameter(), _FakeParameter()]
        self.eval_called = False

    def eval(self) -> None:
        self.eval_called = True

    def parameters(self):
        return self.parameters_list


def test_freeze_model_disables_every_gradient_and_switches_to_eval():
    """Le gel doit passer le modèle en ``eval`` et désactiver tout gradient."""
    model = _FakeFrozenModel()
    harness.freeze_model(model)
    assert model.eval_called
    assert all(parameter.requires_grad is False for parameter in model.parameters_list)


def test_benchmark_source_has_no_training_primitive():
    """Le harnais ne doit contenir aucune primitive d'entraînement."""
    source = _HARNESS_PATH.read_text(encoding="utf-8")
    assert "requires_grad_(True)" not in source
    assert "backward(" not in source
    assert "torch.optim" not in source


def test_benchmark_writes_only_inside_the_repository():
    """La sortie par défaut est stdout ; le cache HF est redirigé dans le dépôt.

    Le test ne doit **pas** exiger l'existence de ``_artefact/`` : ce dossier
    est gitignoré (règle 9 d'`AGENTS.md`) et absent d'un checkout CI. Ce qui
    est vérifié ici, c'est la **redirection** du cache vers l'intérieur du
    dépôt, pas la présence du dossier.
    """
    args = harness.parse_args([])
    assert args.output is None
    assert args.local_files_only is False
    cache_root = Path(harness.os.environ["HF_HOME"]).resolve()
    assert harness._REPO_ROOT.resolve() in cache_root.parents
    assert cache_root != Path.home().resolve()


def test_prefix_cache_geometry_matches_the_opd_scoring():
    """La campagne cache doit couvrir la géométrie réelle du scoring d-OPD.

    Le candidat differe par branche, le suffixe est partage : les suffixes
    testes doivent inclure la longueur de bloc (``N - 1``) et les deux tailles
    de lot de l'App. B (``k`` candidats, ``2k + 1`` avec la reference).
    """
    assert 3 in harness.PREFIX_CACHE_SUFFIXES  # N = 4 : suffixe de 3 tokens
    assert 16 in harness.PREFIX_CACHE_BRANCHES  # k = 16
    assert 33 in harness.PREFIX_CACHE_BRANCHES  # 2k + 1 avec la reference
    assert 1024 in harness.PREFIX_CACHE_CONTEXTS  # plafond de contexte retenu


def test_prefix_cache_campaign_declares_the_fp32_control():
    """Le harnais doit rejouer un cas en FP32 pour distinguer bruit et bug."""
    source = _HARNESS_PATH.read_text(encoding="utf-8")
    assert "prefix_cache_scoring_fp32_control" in source
    assert "torch.float32" in source
