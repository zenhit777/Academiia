"""Tests de la cible future-aware de d-OPD (équations 15-18), CPU uniquement.

Ces tests vérifient des **invariants mathématiques** sur des distributions
jouets. Ils ne mesurent pas la qualité d'un modèle et ne constituent pas une
reproduction du papier.
"""

from __future__ import annotations

import numpy as np
import pytest

from academiia.dopd import (
    REASON_APPLIED,
    REASON_DISABLED,
    REASON_EMPTY_CANDIDATES,
    REASON_FALLBACK,
    REASON_NO_VISIBLE_FUTURE,
    candidate_union,
    compute_future_aware_target,
    relative_future_deltas,
)


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
    np.testing.assert_allclose(result.target, prior, atol=0, rtol=0)


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
    np.testing.assert_allclose(result.target, expected)
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
    np.testing.assert_allclose(result.target, expected)


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
    np.testing.assert_allclose(result.target, expected)


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
    with pytest.raises(ValueError):
        compute_future_aware_target(
            prior, candidate_indices=[0], candidate_loglikelihoods=[np.nan],
            reference_loglikelihood=0.0,
        )
    with pytest.raises(ValueError):
        compute_future_aware_target(
            prior, candidate_indices=[0], candidate_loglikelihoods=[np.inf],
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


def test_relative_deltas_reject_non_finite_reference():
    with pytest.raises(ValueError):
        relative_future_deltas([0.0], np.inf)
    with pytest.raises(ValueError):
        relative_future_deltas([0.0], np.nan)
