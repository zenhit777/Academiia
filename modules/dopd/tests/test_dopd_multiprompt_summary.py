"""Tests CPU du résumé dérivé J15/J16 (aucun torch, aucun GPU).

Le fichier ``results.json`` de J15 est immuable ; le résumé dérivé est le seul
endroit où les corrections J16 (ventilation par signe, TV renormalisée, décision
reproductible) sont calculées. Ces tests vérifient que les chiffres publiés sont
recalculables depuis l'artefact et qu'ils restent ceux annoncés dans le rapport.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "harness" / "summarize_multiprompt.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "dopd_summarize_multiprompt", _MODULE_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
module = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = module
_SPEC.loader.exec_module(module)

_EVAL_DIR = (
    Path(__file__).resolve().parents[1]
    / "eval"
    / "annexe_d_qwen3_0.6b_multiprompt"
)
_RESULTS = _EVAL_DIR / "results.json"
_DERIVED = _EVAL_DIR / "derived_summary.json"


@pytest.fixture(scope="module")
def derived() -> dict:
    return json.loads(_DERIVED.read_text(encoding="utf-8"))


def _assert_close(left, right, *, path: str = "root") -> None:
    if isinstance(right, dict):
        assert isinstance(left, dict), path
        assert set(left) == set(right), path
        for key in right:
            _assert_close(left[key], right[key], path=f"{path}.{key}")
        return
    if isinstance(right, list):
        assert isinstance(left, list), path
        assert len(left) == len(right), path
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            _assert_close(left_item, right_item, path=f"{path}[{index}]")
        return
    if isinstance(right, bool):
        assert left is right, path
        return
    if isinstance(right, (int, float)):
        assert left == pytest.approx(right, rel=1e-12, abs=1e-12), path
        return
    assert left == right, path


def test_derived_summary_is_recomputable_from_results(derived):
    recomputed = module.derive(_RESULTS)
    for key in ("headline", "prompts", "marginals", "top_k", "gate"):
        _assert_close(recomputed[key], derived[key], path=key)


def test_prompt_sign_correction_is_published(derived):
    prompts = derived["prompts"]
    assert prompts["n_prompts"] == 20
    assert prompts["n_positive"] == 17
    assert prompts["n_negative"] == 3
    assert prompts["negative_prompt_ids"] == ["p06", "p10", "p18"]
    assert prompts["n_significant_positive"] == 14
    assert prompts["n_significant_negative"] == 1
    assert prompts["significant_negative_prompt_ids"] == ["p18"]
    assert prompts["proportion_prompts_significant_positive"] == pytest.approx(0.70)
    assert prompts["proportion_prompts_ci_excludes_zero"] == pytest.approx(0.75)
    assert prompts["sign_test_exact_two_sided_p"] == pytest.approx(
        0.0025768280029296875, abs=1e-15
    )


def test_relative_reduction_is_the_multiprompt_value(derived):
    headline = derived["headline"]
    assert headline["mean_causal_kl"] == pytest.approx(0.9350779287713015, abs=1e-12)
    assert headline["mean_future_kl"] == pytest.approx(0.8548873434331475, abs=1e-12)
    assert headline["mean_delta_kl"] == pytest.approx(0.0801905853381537, abs=1e-12)
    assert headline["relative_reduction_pct"] == pytest.approx(8.57581842868697, abs=1e-9)
    # Le 16,12 % de J14 reste consigné comme référence historique étiquetée.
    assert headline["relative_reduction_j14_pct_reference"] == pytest.approx(16.12)


def test_degraded_marginals_are_published(derived):
    marginals = derived["marginals"]
    assert marginals["n_visible_future"] == 35040
    assert marginals["n_degraded"] == 4661
    assert marginals["degraded_fraction"] == pytest.approx(0.13301940639269405)
    assert marginals["mean_delta_kl_degraded"] == pytest.approx(
        -1.186311445398854, abs=1e-12
    )
    assert marginals["min_delta_kl_degraded"] == pytest.approx(
        -28.709425965315273, abs=1e-12
    )
    assert marginals["n_zero_exact"] == 26080
    assert marginals["n_near_zero_numerical"] == 72


def test_top_k_tv_is_corrected_and_legacy_value_kept(derived):
    top_k = derived["top_k"]
    assert top_k["k"] == 16
    assert top_k["n_marginals_measured"] == 61120
    assert top_k["mean_discarded_mass"] == pytest.approx(
        0.029534453987388114, abs=1e-15
    )
    assert top_k["mean_tv_distance_renormalized"] == pytest.approx(
        top_k["mean_discarded_mass"], abs=1e-15
    )
    assert top_k["mean_tv_distance_as_published_j15"] == pytest.approx(
        0.014767226993694075, abs=1e-15
    )
    assert top_k["tv_relation"] == "legacy_half_tv"
    assert top_k["max_discarded_mass"] == pytest.approx(
        0.3731896658778219, abs=1e-15
    )


def test_gate_decision_is_reproducible_and_sign_annotated(derived):
    gate = derived["gate"]
    assert gate["derived_status"] == "GO conditionnel"
    assert gate["published_status_in_results_json"] == "GO conditionnel"
    assert gate["consistent_with_published"] is True
    criteria = gate["criteria"]
    assert criteria["1_aggregate_delta_kl_positive_ci_excludes_zero"]["met"] is True
    assert criteria["2_median_delta_kl_positive"]["met"] is False
    criterion3 = criteria[
        "3_proportion_prompts_significant_positive_ge_threshold"
    ]
    assert criterion3["met"] is True
    assert criterion3["value"] == pytest.approx(0.70)
    assert criterion3["historical_ci_excludes_zero"]["n_prompts"] == 15
    assert (
        criterion3["historical_ci_excludes_zero"]["n_prompts_significant_negative"]
        == 1
    )
    assert criteria["4_sign_test_p_lt_0.05"]["met"] is True
    assert criteria["5_effectif_blocks_ge_min_marginals_ge_min"]["met"] is True
