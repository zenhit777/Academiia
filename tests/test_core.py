"""Tests du code générique (`academiia.core`) — sans GPU, sans réseau.

Le code de `academiia.core` est **figé** : il alimente le verdict G1. Ces
tests verrouillent donc :
  * l'algorithme de l'intervalle de Wilson (valeurs de référence) ;
  * le déterminisme et les propriétés de l'IC bootstrap apparié ;
  * les empreintes SHA-256 et l'empreinte d'index HuggingFace ;
  * la détection de génération dégénérée.

Lancer : pytest tests -q
"""

import hashlib
import json

import pytest

from academiia.core.common import (index_fingerprint, list_source_sidecar_files,
                                   sha256_file)
from academiia.core.stats import bootstrap_ci95, wilson
from academiia.core.text import degenerate


def test_wilson_valeurs_de_reference():
    """0/10 -> borne nulle ; 10/10 -> borne haute à 1."""
    lo, hi = wilson(0, 10)
    assert lo == 0.0
    assert 0.0 < hi < 1.0
    lo, hi = wilson(10, 10)
    assert hi == 1.0
    assert 0.0 < lo < 1.0


def test_wilson_ensemble_vide():
    assert wilson(0, 0) == (None, None)


def test_wilson_large_effectif_plus_fort():
    """L'IC se resserre quand l'effectif grandit (à écart constant)."""
    lo_small, hi_small = wilson(50, 100)
    lo_big, hi_big = wilson(500, 1000)
    assert (hi_big - lo_big) < (hi_small - lo_small)


def test_bootstrap_deterministe_a_graine_fixe():
    diffs = [1, -1, 0, 1, 1, -1, 0, 0, 1, -1] * 3
    assert bootstrap_ci95(diffs, iters=500, seed=7) == \
        bootstrap_ci95(diffs, iters=500, seed=7)


def test_bootstrap_ensemble_vide():
    assert bootstrap_ci95([], iters=100, seed=0) == (None, None)


def test_bootstrap_ecart_positif_icc_contient_le_point_estimate():
    """Sur un écart constant, l'IC est réduit au point (bootstrap sans variance)."""
    lo, hi = bootstrap_ci95([0.2] * 50, iters=200, seed=1)
    assert lo == pytest.approx(0.2)
    assert hi == pytest.approx(0.2)


def test_bootstrap_icc_plus_large_que_zero_quand_ecart_positif():
    diffs = [1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1]
    lo, hi = bootstrap_ci95(diffs, iters=2000, seed=3)
    assert lo > 0.0
    assert hi >= lo


def test_sha256_file_conforme_a_hashlib(tmp_path):
    p = tmp_path / "f.bin"
    payload = b"academiia" * 1000
    p.write_bytes(payload)
    assert sha256_file(str(p)) == hashlib.sha256(payload).hexdigest()


def test_index_fingerprint_compte_tenseurs_et_fichiers(tmp_path):
    index = {
        "metadata": {"total_size": 12},
        "weight_map": {
            "model.a.weight": "shard-1.safetensors",
            "model.b.weight": "shard-1.safetensors",
            "model.c.weight": "shard-2.safetensors",
        },
    }
    path = tmp_path / "model.safetensors.index.json"
    path.write_text(json.dumps(index))
    fp = index_fingerprint(str(path))
    assert fp["n_tensors"] == 3
    assert fp["n_files"] == 2
    assert len(fp["weight_map_sha256"]) == 16


def test_sidecars_excluent_poids_index_et_readme(tmp_path):
    for name in ("model-00001-of-00001.safetensors", "model.safetensors.index.json",
                 "README.md", "tokenizer.json", "config.json", ".hidden"):
        (tmp_path / name).write_text("x")
    sidecars = list_source_sidecar_files(str(tmp_path))
    assert sidecars == ["config.json", "tokenizer.json"]


def test_degenerate_detecte_vocabulaire_repetitif():
    assert degenerate("a " * 40) is not None


def test_degenerate_detecte_repetition_de_bloc():
    assert degenerate("abcd" * 30) is not None


def test_degenerate_ne_signale_pas_un_texte_normal():
    assert degenerate("Le chat dort sur le tapis, il ronronne doucement.") is None


def test_degenerate_signale_un_texte_vide():
    assert degenerate("") == "texte vide"
