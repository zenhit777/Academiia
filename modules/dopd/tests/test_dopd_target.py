"""Tests de la cible future-aware de d-OPD (équations 15-18), CPU uniquement.

Ces tests vérifient des **invariants mathématiques** sur des distributions
jouets. Ils ne mesurent pas la qualité d'un modèle et ne constituent pas une
reproduction du papier.

Le dtype d'arithmétique par défaut est ``float32`` (App. B du papier). Les
comparaisons à une forme close en float64 utilisent donc une tolérance
adaptée au float32 ; le chemin ``float64`` est testé séparément à haute
précision.
"""

from __future__ import annotations

import numpy as np
import pytest

from academiia.dopd import (
    REASON_APPLIED,
    REASON_DISABLED,
    REASON_EMPTY_CANDIDATES,
    REASON_FALLBACK,
    REASON_NON_FINITE_SCORES,
    REASON_NO_VISIBLE_FUTURE,
    candidate_union,
    compute_future_aware_target,
    relative_future_deltas,
)

# Tolérance des comparaisons calculées en float32 (dtype du papier).
FLOAT32_RTOL = 1e-6


def _normalized(prior: np.ndarray) -> np.ndarray:
    """Prior renormalisé : c'est la forme exacte que le fallback restitue.

    La renormalisation ne change le prior que de son dernier bit (une somme
    float64 de probabilités n'est pas exactement 1), mais elle garantit que
    la cible somme 1 — invariant annoncé par ``TargetResult``.
    """
    return prior / prior.sum()


def test_no_visible_future_returns_causal_prior():
    prior = np.array([0.4, 0.3, 0.2, 0.1])
    result = compute_future_aware_target(
        prior,
        candidate_indices=[1, 2],
        candidate_loglikelihoods=[-3.0, -2.0],
        reference_loglikelihood=-3.0,
        visible_future=False,
    )
    assert result.applied is False
    assert result.reason == REASON_NO_VISIBLE_FUTURE
    assert result.n_candidates == 0
    np.testing.assert_allclose(result.target, _normalized(prior), atol=0, rtol=0)


def test_disabled_gate_returns_causal_prior():
    prior = np.array([0.5, 0.5])
    result = compute_future_aware_target(prior, apply=False)
    assert result.applied is False
    assert result.reason == REASON_DISABLED
    np.testing.assert_allclose(result.target, prior, atol=0, rtol=0)


def test_empty_candidate_set_returns_causal_prior():
    prior = np.array([0.5, 0.5])
    result = compute_future_aware_target(
        prior,
        candidate_indices=[],
        candidate_loglikelihoods=[],
        reference_loglikelihood=-1.0,
    )
    assert result.applied is False
    assert result.reason == REASON_EMPTY_CANDIDATES
    np.testing.assert_allclose(result.target, prior, atol=0, rtol=0)


def test_reference_token_has_zero_delta():
    deltas = relative_future_deltas(np.array([-4.0, -2.0, -1.0]), -2.0)
    assert deltas[1] == 0.0
    np.testing.assert_allclose(deltas, [-2.0, 0.0, 1.0])


def test_manual_reweighting_matches_equation_18():
    prior = np.array([0.4, 0.3, 0.2, 0.1])
    # Candidats 1 et 2 ; le candidat 2 a un suffixe deux fois plus probable
    # que la reference, le candidat 1 autant que la reference.
    reference = -5.0
    scores = np.array([-5.0, -5.0 + np.log(2.0)])
    result = compute_future_aware_target(
        prior,
        candidate_indices=[1, 2],
        candidate_loglikelihoods=scores,
        reference_loglikelihood=reference,
    )
    expected_weights = np.array([0.4, 0.3, 0.4, 0.1])
    expected = expected_weights / expected_weights.sum()
    assert result.applied is True
    assert result.reason == REASON_APPLIED
    assert result.n_candidates == 2
    np.testing.assert_allclose(result.target, expected, rtol=FLOAT32_RTOL)
    assert result.target.sum() == pytest.approx(1.0)


def test_tokens_outside_candidates_keep_unit_factor():
    prior = np.array([0.4, 0.3, 0.2, 0.1])
    result = compute_future_aware_target(
        prior,
        candidate_indices=[3],
        candidate_loglikelihoods=[np.log(3.0)],
        reference_loglikelihood=0.0,
    )
    expected_weights = np.array([0.4, 0.3, 0.2, 0.3])
    expected = expected_weights / expected_weights.sum()
    np.testing.assert_allclose(result.target, expected, rtol=FLOAT32_RTOL)


def test_full_vocabulary_correction_matches_closed_form():
    prior = np.array([0.1, 0.2, 0.3, 0.4])
    scores = np.array([-2.0, -1.5, -0.5, 0.0])
    reference = -1.5
    result = compute_future_aware_target(
        prior,
        candidate_indices=np.arange(4),
        candidate_loglikelihoods=scores,
        reference_loglikelihood=reference,
    )
    weights = prior * np.exp(scores - reference)
    expected = weights / weights.sum()
    np.testing.assert_allclose(result.target, expected, rtol=FLOAT32_RTOL)


def test_float64_path_matches_closed_form_to_high_precision():
    """Le chemin float64 est la référence de précision maximale."""
    prior = np.array([0.1, 0.2, 0.3, 0.4])
    scores = np.array([-2.0, -1.5, -0.5, 0.0])
    result = compute_future_aware_target(
        prior,
        candidate_indices=np.arange(4),
        candidate_loglikelihoods=scores,
        reference_loglikelihood=-1.5,
        arithmetic_dtype="float64",
    )
    weights = prior * np.exp(scores + 1.5)
    expected = weights / weights.sum()
    np.testing.assert_allclose(result.target, expected, atol=1e-15, rtol=1e-12)


def test_shift_invariance_of_all_future_scores():
    prior = np.array([0.2, 0.3, 0.5])
    scores = np.array([-2.0, -1.0])
    reference = -2.0
    base = compute_future_aware_target(
        prior,
        candidate_indices=[0, 2],
        candidate_loglikelihoods=scores,
        reference_loglikelihood=reference,
    )
    shifted = compute_future_aware_target(
        prior,
        candidate_indices=[0, 2],
        candidate_loglikelihoods=scores + 100.0,
        reference_loglikelihood=reference + 100.0,
    )
    np.testing.assert_allclose(base.target, shifted.target, atol=1e-12)


def test_exact_zeros_stay_zero():
    prior = np.array([0.5, 0.5, 0.0])
    result = compute_future_aware_target(
        prior,
        candidate_indices=[0, 2],
        candidate_loglikelihoods=[0.0, 10.0],
        reference_loglikelihood=0.0,
    )
    assert result.target[2] == 0.0
    assert result.target.sum() == pytest.approx(1.0)


def test_impossible_candidate_suffix_gets_zero_weight():
    """Un delta -inf est légitime : le poids du candidat s'annule."""
    prior = np.array([0.5, 0.5])
    result = compute_future_aware_target(
        prior,
        candidate_indices=[0],
        candidate_loglikelihoods=[-np.inf],
        reference_loglikelihood=0.0,
    )
    assert result.applied is True
    np.testing.assert_allclose(result.target, [0.0, 1.0])


def test_non_finite_normalizer_falls_back_to_causal_prior():
    """Tous les poids nuls : normaliseur nul -> fallback causal (App. B)."""
    prior = np.array([1.0, 0.0])
    result = compute_future_aware_target(
        prior,
        candidate_indices=[0],
        candidate_loglikelihoods=[-np.inf],
        reference_loglikelihood=0.0,
    )
    assert result.applied is False
    assert result.reason == REASON_FALLBACK
    assert result.n_candidates == 1
    np.testing.assert_allclose(result.target, prior)


def test_overflowing_correction_factor_falls_back_to_causal_prior():
    """Un delta fini assez grand déborde en float32 : pas d'écrêtage, fallback."""
    prior = np.array([0.5, 0.5])
    result = compute_future_aware_target(
        prior,
        candidate_indices=[0],
        candidate_loglikelihoods=[1000.0],
        reference_loglikelihood=0.0,
    )
    assert result.applied is False
    assert result.reason == REASON_FALLBACK
    np.testing.assert_allclose(result.target, prior)


def test_underflowing_correction_falls_back_to_causal_prior():
    """Tous les facteurs sous-débitent à zéro : normaliseur nul -> fallback."""
    prior = np.array([0.5, 0.5])
    result = compute_future_aware_target(
        prior,
        candidate_indices=[0, 1],
        candidate_loglikelihoods=[-1000.0, -1001.0],
        reference_loglikelihood=0.0,
    )
    assert result.applied is False
    assert result.reason == REASON_FALLBACK
    np.testing.assert_allclose(result.target, prior)


def test_float64_path_does_not_overflow_where_float32_does():
    """Le dtype d'arithmétique change le seuil de débordement, pas la règle."""
    prior = np.array([0.5, 0.5])
    kwargs = dict(
        candidate_indices=[0],
        candidate_loglikelihoods=[100.0],
        reference_loglikelihood=0.0,
    )
    in_float32 = compute_future_aware_target(prior, **kwargs)
    in_float64 = compute_future_aware_target(prior, arithmetic_dtype="float64", **kwargs)
    assert in_float32.applied is False
    assert in_float32.reason == REASON_FALLBACK
    assert in_float64.applied is True
    # exp(100) reste fini en float64 : le candidat domine sans être infini.
    assert in_float64.target[0] == pytest.approx(1.0, abs=1e-12)
    assert in_float64.target[1] < 1e-40


@pytest.mark.parametrize("bad_score", [np.nan, np.inf])
def test_non_finite_candidate_score_falls_back_to_causal_prior(bad_score):
    """Un score candidat non fini rend le normaliseur non fini : fallback."""
    prior = np.array([0.5, 0.5])
    result = compute_future_aware_target(
        prior,
        candidate_indices=[0, 1],
        candidate_loglikelihoods=[bad_score, -1.0],
        reference_loglikelihood=0.0,
    )
    assert result.applied is False
    assert result.reason == REASON_NON_FINITE_SCORES
    np.testing.assert_allclose(result.target, prior)


@pytest.mark.parametrize("bad_reference", [np.nan, np.inf, -np.inf])
def test_non_finite_reference_falls_back_to_causal_prior(bad_reference):
    """Une référence non finie : le ratio exp(Δs) n'a pas de sens, fallback."""
    prior = np.array([0.5, 0.5])
    result = compute_future_aware_target(
        prior,
        candidate_indices=[0],
        candidate_loglikelihoods=[-1.0],
        reference_loglikelihood=bad_reference,
    )
    assert result.applied is False
    assert result.reason == REASON_NON_FINITE_SCORES
    np.testing.assert_allclose(result.target, prior)


def test_near_one_prior_is_renormalised():
    """Un prior accepté mais non exact est renormalisé : la cible somme 1."""
    prior = np.array([0.5, 0.5000000000005])
    result = compute_future_aware_target(prior, apply=False)
    assert result.target.sum() == pytest.approx(1.0, abs=1e-15)
    # La renormalisation retire l'écart de 5e-13 : la cible est le prior
    # normalisé, pas le prior brut.
    np.testing.assert_allclose(result.target, _normalized(prior), atol=0, rtol=0)
    assert result.target[0] != prior[0]


def test_candidate_union_is_sorted_and_deduplicated():
    union = candidate_union([3, 1, 2], [2, 5])
    np.testing.assert_array_equal(union, [1, 2, 3, 5])
    assert candidate_union([], []).size == 0


@pytest.mark.parametrize(
    "prior, exc",
    [
        (np.array([0.5, -0.5]), ValueError),
        (np.array([0.5, 0.4]), ValueError),  # ne somme pas a 1
        (np.array([0.0, 0.0]), ValueError),
        (np.array([[0.5, 0.5]]), ValueError),
        (np.array([np.nan, 0.5]), ValueError),
        (np.array([np.inf, 0.5]), ValueError),
    ],
)
def test_invalid_prior_rejected(prior, exc):
    with pytest.raises(exc):
        compute_future_aware_target(prior, apply=False)


def test_invalid_candidate_and_scores_rejected():
    prior = np.array([0.5, 0.5])
    with pytest.raises(IndexError):
        compute_future_aware_target(
            prior, candidate_indices=[2], candidate_loglikelihoods=[0.0],
            reference_loglikelihood=0.0,
        )
    with pytest.raises(ValueError):
        compute_future_aware_target(
            prior, candidate_indices=[0, 0], candidate_loglikelihoods=[0.0, 0.0],
            reference_loglikelihood=0.0,
        )
    with pytest.raises(TypeError):
        compute_future_aware_target(
            prior, candidate_indices=[0.0], candidate_loglikelihoods=[0.0],
            reference_loglikelihood=0.0,
        )
    with pytest.raises(ValueError):
        compute_future_aware_target(
            prior, candidate_indices=[0], candidate_loglikelihoods=[0.0, 1.0],
            reference_loglikelihood=0.0,
        )


def test_missing_required_inputs_rejected():
    prior = np.array([0.5, 0.5])
    with pytest.raises(ValueError):
        compute_future_aware_target(prior, reference_loglikelihood=0.0)
    with pytest.raises(ValueError):
        compute_future_aware_target(
            prior, candidate_indices=[0], candidate_loglikelihoods=[0.0]
        )
    with pytest.raises(ValueError):
        compute_future_aware_target(
            prior, candidate_indices=[0], reference_loglikelihood=0.0
        )


def test_unknown_arithmetic_dtype_rejected():
    prior = np.array([0.5, 0.5])
    with pytest.raises(ValueError):
        compute_future_aware_target(
            prior,
            candidate_indices=[0],
            candidate_loglikelihoods=[0.0],
            reference_loglikelihood=0.0,
            arithmetic_dtype="float16",
        )
    with pytest.raises(ValueError):
        relative_future_deltas([0.0], 0.0, arithmetic_dtype="bfloat16")


def test_relative_deltas_propagate_non_finite_reference():
    """Le helper propage les valeurs non finies ; la cible décide du fallback."""
    with np.errstate(invalid="ignore"):
        # s(v) - s(x̂) avec s(x̂) = +inf : le delta vaut -inf.
        assert np.isneginf(relative_future_deltas([0.0], np.inf)).all()
        assert np.isnan(relative_future_deltas([0.0], np.nan)).all()
