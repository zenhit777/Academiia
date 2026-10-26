"""Tests CPU de la sémantique du harnais de décision J18.

Le cœur décisionnel (table §4 de J18_PRE, test des signes, assemblage d'un
marginal sur un support réel) est testé sans GPU : un fake modèle et un fake
professeur (duck-typed) suffisent pour `_build_restricted_support` et
`_marginal_for_position`. `torch` est lié par `importorskip`, pas par
`import torch`.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
import numpy as np
import pytest

# Garde de collecte : sans torch, le module entier est ignoré (et non une erreur
# d'import).
torch = pytest.importorskip("torch", reason="torch absent : tests J18 ignores")

_HARNESS_PATH = Path(__file__).resolve().parents[1] / "harness" / "run_j18_decision.py"
_SPEC = importlib.util.spec_from_file_location("dopd_harness_j18_decision", _HARNESS_PATH)
assert _SPEC is not None and _SPEC.loader is not None
j18 = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = j18
_SPEC.loader.exec_module(j18)


# ---------------------------------------------------------------------------
# Test des signes (binomiale exacte bilatérale)
# ---------------------------------------------------------------------------


def test_binomial_p_value_is_small_for_a_strong_effect() -> None:
    """3000 signes positifs pour 200 négatifs : p quasi nulle."""
    p = j18._binominal_two_tailed_p_value(3000, 200)
    assert p < 1e-6


def test_binomial_p_value_near_one_for_a_balanced_effect() -> None:
    """51 positifs pour 49 : aucun signe, p proche de 1."""
    p = j18._binominal_two_tailed_p_value(51, 49)
    assert 0.8 < p <= 1.0


def test_binomial_p_value_is_nan_without_effective_samples() -> None:
    """Aucun marginal non nul : valeur en p non définie."""
    assert j18._binominal_two_tailed_p_value(0, 0) != j18._binominal_two_tailed_p_value(0, 0)


# ---------------------------------------------------------------------------
# Table de décision (§4 de J18_PRE)
# ---------------------------------------------------------------------------


def _statistics(
    *,
    mean_delta_kl: float,
    median_delta_kl: float,
    ci_lower: float,
    n_prompts: int = 13,
    n_excludes_zero: int = 10,
    positive: int = 3000,
    negative: int = 200,
) -> dict:
    return {
        "mean_delta_kl": mean_delta_kl,
        "median_delta_kl": median_delta_kl,
        "aggregate_ci95_delta_kl": [ci_lower, ci_lower + 0.06],
        "n_prompts": n_prompts,
        "n_prompts_ci_excludes_zero": n_excludes_zero,
        "sign_test": {"positive": positive, "negative": negative, "n": positive + negative},
    }


def _gate(**overrides) -> dict:
    kwargs = dict(
        blocks_per_configuration={"N4_V16": 120},
        total_marginals=4680,
        numerical_integrity_met=True,
        manifest_present=True,
        teacher_frozen_met=True,
        model_source_in_repo=True,
        no_regression_met=True,
    )
    kwargs.update(overrides)
    stats_kwargs = {key: value for key, value in overrides.items() if key in {
        "mean_delta_kl",
        "median_delta_kl",
        "ci_lower",
        "n_prompts",
        "n_excludes_zero",
        "positive",
        "negative",
    }}
    return j18._decide_j18_gate(
        statistics=_statistics(**stats_kwargs),
        sign_p_value=0.001,
        **{key: value for key, value in kwargs.items() if key not in {
            "mean_delta_kl",
            "median_delta_kl",
            "ci_lower",
            "n_prompts",
            "n_excludes_zero",
            "positive",
            "negative",
        }},
    )


def test_decision_is_go_when_all_criteria_are_met() -> None:
    """C1-C5 + C6-C10 tenus : GO pour ouvrir un StudentPolicy réduit en J19."""
    decision = _gate(mean_delta_kl=0.05, median_delta_kl=0.01, ci_lower=0.02)
    assert decision["status"] == "GO"
    assert decision["reserves"] == []


def test_decision_is_no_go_when_primary_ci_includes_zero() -> None:
    """C1 non tenu (IC 95 % inclut zéro) : NO-GO, sans autre calcul."""
    decision = _gate(mean_delta_kl=0.01, median_delta_kl=0.02, ci_lower=-0.01)
    assert decision["status"] == "NO-GO"
    assert not decision["criteria"]["C1_aggregate_delta_kl_positive_ci95_excludes_zero"]["met"]


def test_decision_is_go_conditionnel_when_median_is_non_positive() -> None:
    """C2 fragile (médiane <= 0) mais C1 tenu : GO conditionnel avec réserve."""
    decision = _gate(mean_delta_kl=0.05, median_delta_kl=0.0, ci_lower=0.02)
    assert decision["status"] == "GO conditionnel"
    assert any("médiane" in reserve for reserve in decision["reserves"])


def test_decision_is_go_conditionnel_when_process_criterion_fails() -> None:
    """C10 non-régression échoué : décision explicite, jamais un GO silencieux."""
    decision = _gate(
        mean_delta_kl=0.05, median_delta_kl=0.01, ci_lower=0.02, no_regression_met=False
    )
    assert decision["status"] != "GO"
    assert any("C10" in reserve for reserve in decision["reserves"])
    assert decision["process_criteria"]["C10_no_regression"]["met"] is False


def test_decision_is_go_conditionnel_when_effect_carried_by_few_prompts() -> None:
    """C3 fragile (proportion < 0,5) : GO conditionnel avec réserve nommée."""
    decision = _gate(
        mean_delta_kl=0.05, median_delta_kl=0.01, ci_lower=0.02, n_excludes_zero=4
    )
    assert decision["status"] == "GO conditionnel"
    assert any("proportion" in reserve for reserve in decision["reserves"])


# ---------------------------------------------------------------------------
# Assemblage d'un marginal sur un support réel
# ---------------------------------------------------------------------------


class _ToyConfig:
    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size


class _FakeOutput:
    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits


class _FakeModel:
    """Modèle déterministe : logits dépendant du dernier token (pas de cache)."""

    def __init__(self, vocab_size: int) -> None:
        self.config = _ToyConfig(vocab_size)

    def __call__(self, input_ids: torch.Tensor, **_: object) -> _FakeOutput:
        batch, length = input_ids.shape
        base = torch.arange(self.config.vocab_size, dtype=torch.float32)
        # Distribution non uniforme : probabilité décroissante avec l'identifiant.
        logits = (base.reshape(1, 1, -1) - input_ids[:, -1:, None] * 0.1)
        return _FakeOutput(logits)


class _FakeScorer:
    """Professeur duck-typed : prior causal + scores de suffixe contrôlés."""

    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size

    def causal_prior(self, prefix: tuple[int, ...]) -> np.ndarray:
        base = np.arange(1, self.vocab_size + 1, dtype=np.float64)
        prior = 1.0 / base
        return prior / prior.sum()

    def suffix_loglikelihood(
        self,
        prefix_ids: tuple[int, ...],
        candidate_ids: tuple[int, ...],
        suffix_ids: tuple[int, ...],
    ) -> np.ndarray:
        # Score décroissant avec l'identifiant : le futur repondère les candidats.
        return np.array(
            [1.0 / (1.0 + float(token)) for token in candidate_ids], dtype=np.float64
        )


def _toy_support(vocab_size: int = 8, block_size: int = 4, width: int = 3) -> object:
    support = j18._build_restricted_support(
        _FakeModel(vocab_size),
        torch.tensor([1, 2, 3][:block_size - 1], dtype=torch.long),
        block_size,
        width,
        torch.device("cpu"),
    )
    return support


def test_marginal_for_position_computes_finite_delta_kl() -> None:
    """Un marginal à futur visible rend des KL finies et renormalisées."""
    support = _toy_support()
    sequences, log_joint = support.enumerate_joint()
    prompt_ids = (1, 2, 3)
    # Une ligne réelle du joint = un bloc complété x̂.
    block = tuple(int(token) for token in sequences[0].tolist())
    record = j18._marginal_for_position(
        _FakeScorer(support.supports[0].max().item() + 1),
        prompt_ids,
        block,
        0,
        support,
        sequences,
        log_joint,
        0,
        "p01",
    )
    assert record is not None
    for key in ("causal_kl", "future_kl", "delta_kl"):
        assert np.isfinite(record[key])
    # Cohérence interne : ΔKL = KL(causal) − KL(future).
    assert record["delta_kl"] == pytest.approx(
        record["causal_kl"] - record["future_kl"], rel=1e-6
    )
    assert record["max_sum_error"] <= 1e-5
    assert record["prompt"] == "p01"
    assert record["block_index"] == 0


def test_marginal_without_visible_future_returns_none() -> None:
    """À la dernière position (futur nul), aucun marginal admissible."""
    support = _toy_support()
    sequences, log_joint = support.enumerate_joint()
    block = tuple(int(token) for token in sequences[0].tolist())
    record = j18._marginal_for_position(
        _FakeScorer(8),
        (1, 2, 3),
        block,
        len(block) - 1,
        support,
        sequences,
        log_joint,
        0,
        "p01",
    )
    assert record is None


def test_mask_positions_always_have_a_visible_future() -> None:
    """Chaque position de MASK_POSITIONS a un suffixe non vide (N=4)."""
    for position in j18.MASK_POSITIONS:
        suffix_length = j18.BLOCK_SIZE - 1 - position
        assert suffix_length >= j18.SUFFIX_LENGTH_MIN


def test_support_width_equals_candidate_k() -> None:
    """La largeur du support vaut k (=16) : exact et cibles sur le même support."""
    assert j18.VOCAB_WIDTH == j18.K_CANDIDATES == 16


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
