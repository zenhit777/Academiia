"""Tests CPU du noyau numérique du diagnostic Annexe D (J14).

Ces tests vérifient les invariants mathématiques de l'énumération exacte sur
des supports minuscules. Ils ne chargent aucun modèle et ne constituent pas
une preuve de la porte scientifique ; cette preuve vient du résultat GPU
versionné.
"""

from __future__ import annotations

import numpy as np
import pytest

from academiia.dopd.diagnostic import (
    RestrictedSupport,
    bootstrap_delta_ci,
    decide_multiprompt_gate,
    evaluate_gate,
    exact_token_posteriors,
    exact_token_posteriors_reference,
    kl_exact_to_target,
    prompt_sign_breakdown,
    sample_compatible_index,
    summarize_marginals,
    truncate_causal_gap,
    verify_enumeration_agreement,
)


def _toy_support() -> RestrictedSupport:
    """Support 2 positions x 2 tokens avec des probabilités connues."""
    return RestrictedSupport(
        supports=(np.asarray([10, 11]), np.asarray([20, 21])),
        prefixes=(((),), ((10,), (11,))),
        conditional_log_probs=(
            np.asarray([[np.log(0.6), np.log(0.4)]]),
            np.asarray(
                [
                    [np.log(0.8), np.log(0.2)],
                    [np.log(0.3), np.log(0.7)],
                ]
            ),
        ),
    )


def test_restricted_support_rejects_misaligned_prefixes():
    with pytest.raises(ValueError, match="prefixes"):
        RestrictedSupport(
            supports=(np.asarray([10, 11]), np.asarray([20, 21])),
            prefixes=(((),), ((10,),)),
            conditional_log_probs=(
                np.asarray([[np.log(0.6), np.log(0.4)]]),
                np.asarray([[np.log(0.8), np.log(0.2)]]),
            ),
        )


def test_enumerate_joint_is_normalized_and_factorizes():
    support = _toy_support()
    sequences, log_joint = support.enumerate_joint()
    assert sequences.shape == (4, 2)
    probabilities = np.exp(log_joint)
    np.testing.assert_allclose(probabilities.sum(), 1.0, atol=1e-12)
    # (10,20) : 0.6*0.8 ; (11,21) : 0.4*0.7
    np.testing.assert_allclose(probabilities[0], 0.48, atol=1e-12)
    np.testing.assert_allclose(probabilities[3], 0.28, atol=1e-12)


def test_causal_prior_renormalizes_restricted_support():
    support = _toy_support()
    prior = support.causal_prior((10,), 1)
    np.testing.assert_allclose(prior, [0.8, 0.2], atol=1e-12)
    assert float(prior.sum()) == pytest.approx(1.0, abs=1e-12)


def test_exact_posterior_conditions_on_visible_token():
    support = _toy_support()
    sequences, log_joint = support.enumerate_joint()
    posterior = exact_token_posteriors(
        sequences,
        log_joint,
        visible={0: 10},
        supports=support.supports,
    )
    # P(x1=20 | x0=10) = 0.8 ; P(x1=21 | x0=10) = 0.2
    np.testing.assert_allclose(posterior[1], [0.8, 0.2], atol=1e-12)


def test_two_enumeration_implementations_agree():
    support = _toy_support()
    sequences, log_joint = support.enumerate_joint()
    visible = {0: 11}
    vectorized = exact_token_posteriors(sequences, log_joint, visible, support.supports)
    reference = exact_token_posteriors_reference(
        sequences, log_joint, visible, support.supports
    )
    np.testing.assert_allclose(vectorized[1], reference[1], atol=1e-12, rtol=0.0)
    assert (
        verify_enumeration_agreement(sequences, log_joint, visible, support.supports)
        <= 1e-12
    )


def test_exact_posterior_sums_over_all_compatible_completions():
    """Posterior exact avec deux positions masquées, calculé à la main.

    Support : P(x0=10)=0,6, P(x1=20|x0=10)=0,8, P(x1=20|x0=11)=0,3.
    Avec rien de visible, le posterior de chaque position est la marginale
    de la loi jointe : P(x0=10)=0,6 et P(x1=20)=0,48+0,12=0,60.
    """
    support = _toy_support()
    sequences, log_joint = support.enumerate_joint()
    posterior = exact_token_posteriors(
        sequences, log_joint, visible={}, supports=support.supports
    )
    np.testing.assert_allclose(posterior[0], [0.6, 0.4], atol=1e-12)
    np.testing.assert_allclose(posterior[1], [0.6, 0.4], atol=1e-12)
    reference = exact_token_posteriors_reference(
        sequences, log_joint, visible={}, supports=support.supports
    )
    for position in posterior:
        np.testing.assert_allclose(posterior[position], reference[position], atol=1e-12)


def test_exact_posterior_excludes_tokens_outside_the_support():
    """Un token absent du support ne reçoit aucune masse et ne contamine pas."""
    sequences = np.asarray([[10, 20], [11, 21]], dtype=np.int64)
    log_joint = np.log(np.asarray([0.6, 0.4]))
    posterior = exact_token_posteriors(
        sequences,
        log_joint,
        visible={0: 11},
        supports=(np.asarray([10, 11]), np.asarray([20, 21])),
    )
    assert float(posterior[1].sum()) == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_allclose(posterior[1], [0.0, 1.0], atol=1e-12)


def test_kl_matches_closed_form():
    exact = np.asarray([0.5, 0.5])
    target = np.asarray([0.25, 0.75])
    expected = 0.5 * np.log(0.5 / 0.25) + 0.5 * np.log(0.5 / 0.75)
    assert kl_exact_to_target(exact, target) == pytest.approx(expected, abs=1e-12)


def test_kl_rejects_zero_target_under_positive_mass():
    with pytest.raises(ValueError, match="nulle"):
        kl_exact_to_target(np.asarray([0.5, 0.5]), np.asarray([0.0, 1.0]))


def test_sample_compatible_index_respects_mask():
    log_weights = np.log(np.asarray([0.1, 0.7, 0.2]))
    compatible = np.asarray([True, True, False])
    rng = np.random.default_rng(7)
    for _ in range(20):
        assert sample_compatible_index(log_weights, compatible, rng) in (0, 1)


def _gate_records(
    *,
    config: str = "N4_V6",
    blocks: int = 100,
    noises: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5),
    causal: float = 1.0,
    future: float = 0.5,
) -> list[dict]:
    """Marginaux synthétiques : un bloc de base par (config, bloc, bruit)."""
    return [
        {
            "config_index": config,
            "block_index": block_index,
            "position": 0,
            "noise": noise,
            "causal_kl": causal,
            "future_kl": future,
        }
        for block_index in range(blocks)
        for noise in noises
    ]


def test_summarize_marginals_computes_paired_delta():
    summary = summarize_marginals(_gate_records(blocks=2, noises=(0.0,)))
    assert summary["n"] == 2
    assert summary["mean_causal_kl"] == pytest.approx(1.0)
    assert summary["mean_future_kl"] == pytest.approx(0.5)
    assert summary["mean_delta_kl"] == pytest.approx(0.5)


def test_bootstrap_ci_excludes_zero_when_effect_is_constant():
    records = _gate_records(blocks=100)
    lower, upper = bootstrap_delta_ci(records, seed=1, iterations=500)
    assert lower == pytest.approx(0.5)
    assert upper == pytest.approx(0.5)


def test_gate_passes_when_primary_criterion_and_counts_are_met():
    decision = evaluate_gate(
        _gate_records(blocks=100),
        seed=1,
        iterations=500,
        minimum_marginals=400,
        minimum_blocks_per_configuration=100,
    )
    assert decision.status == "passed"
    assert decision.primary_ci_excludes_zero
    assert decision.minimum_marginals_met
    assert not decision.reserve_required


def test_gate_fails_when_a_configuration_has_too_few_blocks():
    decision = evaluate_gate(
        _gate_records(blocks=99),
        seed=1,
        iterations=500,
        minimum_marginals=100,
        minimum_blocks_per_configuration=100,
    )
    assert decision.status == "failed"
    assert not decision.minimum_marginals_met


def test_gate_fails_when_total_marginals_are_insufficient():
    decision = evaluate_gate(
        _gate_records(blocks=100),
        seed=1,
        iterations=500,
        minimum_marginals=2000,
        minimum_blocks_per_configuration=100,
    )
    assert decision.status == "failed"
    assert not decision.minimum_marginals_met


def test_gate_fails_when_primary_ci_contains_zero():
    records = _gate_records(blocks=100)
    for index, record in enumerate(records):
        record["future_kl"] = record["causal_kl"] - (0.2 if index % 2 else -0.2)
    decision = evaluate_gate(
        records,
        seed=1,
        iterations=500,
        minimum_marginals=400,
        minimum_blocks_per_configuration=100,
    )
    assert decision.status == "failed"
    assert not decision.primary_ci_excludes_zero


def test_gate_reports_reserve_for_a_degrading_noise_level():
    records = _gate_records(blocks=100)
    for record in records:
        if record["noise"] == 1.5:
            record["future_kl"] = record["causal_kl"] + 0.1
    decision = evaluate_gate(
        records,
        seed=1,
        iterations=500,
        minimum_marginals=400,
        minimum_blocks_per_configuration=100,
    )
    assert decision.status == "passed_with_reserve"
    assert decision.reserve_required
    assert decision.reserve_noise_levels == ("1.5",)


def test_gate_reports_reserve_for_a_degrading_configuration():
    records = _gate_records(config="N4_V6", blocks=100)
    records += _gate_records(config="N6_V6", blocks=100, future=1.1)
    decision = evaluate_gate(
        records,
        seed=1,
        iterations=500,
        minimum_marginals=800,
        minimum_blocks_per_configuration=100,
    )
    assert decision.status == "passed_with_reserve"
    assert decision.reserve_configurations == ("N6_V6",)


def test_gate_fails_on_numerical_integrity_violation():
    decision = evaluate_gate(
        _gate_records(blocks=100),
        seed=1,
        iterations=500,
        minimum_marginals=400,
        minimum_blocks_per_configuration=100,
        numerical_max_error=1e-6,
    )
    assert decision.status == "failed"
    assert not decision.numeric_integrity_met


# ---------------------------------------------------------------------------
# J16 : TV contre la troncature renormalisée, signes et décision reproductible
# ---------------------------------------------------------------------------


def test_truncate_causal_gap_renormalizes_before_tv():
    """TV contre la troncature renormalisée = masse discardée (et non sa moitié)."""
    full = np.asarray([0.5, 0.3, 0.1, 0.1])
    gap = truncate_causal_gap(full, 2)
    assert gap["discarded_mass"] == pytest.approx(0.2, abs=1e-15)
    assert gap["truncated_mass"] == pytest.approx(0.8, abs=1e-15)
    # Troncature renormalisée : [0.625, 0.375, 0, 0] -> TV = 0.2.
    assert gap["tv_distance"] == pytest.approx(0.2, abs=1e-15)
    renormalized = np.asarray([0.625, 0.375, 0.0, 0.0])
    assert gap["tv_distance"] == pytest.approx(
        0.5 * np.abs(full - renormalized).sum(), abs=1e-15
    )
    # La version J15 (non renormalisée, divisée par deux) valait 0.1.
    assert gap["tv_distance"] != pytest.approx(0.1, abs=1e-12)


def test_truncate_causal_gap_no_truncation():
    gap = truncate_causal_gap(np.asarray([0.4, 0.6]), 5)
    assert gap == {"discarded_mass": 0.0, "truncated_mass": 1.0, "tv_distance": 0.0}


def test_truncate_causal_gap_rejects_non_positive_k():
    with pytest.raises(ValueError, match="strictement positif"):
        truncate_causal_gap(np.asarray([0.5, 0.5]), 0)


def _prompt_stats() -> tuple[dict[str, dict], dict[str, tuple[float, float]]]:
    per_prompt = {
        "pos_sig": {"mean_delta_kl": 0.2},
        "pos_not_sig": {"mean_delta_kl": 0.01},
        "neg_sig": {"mean_delta_kl": -0.3},
        "neg_not_sig": {"mean_delta_kl": -0.01},
    }
    prompt_ci = {
        "pos_sig": (0.1, 0.3),
        "pos_not_sig": (-0.1, 0.2),
        "neg_sig": (-0.5, -0.1),
        "neg_not_sig": (-0.2, 0.1),
    }
    return per_prompt, prompt_ci


def test_prompt_sign_breakdown_distinguishes_direction_and_significance():
    per_prompt, prompt_ci = _prompt_stats()
    breakdown = prompt_sign_breakdown(per_prompt, prompt_ci)
    assert breakdown["n_positive"] == 2
    assert breakdown["n_negative"] == 2
    assert breakdown["n_significant_positive"] == 1
    assert breakdown["n_significant_negative"] == 1
    assert breakdown["n_not_significant"] == 2
    assert breakdown["significant_negative_prompt_ids"] == ["neg_sig"]


def _decide(**overrides) -> dict:
    values = {
        "mean_delta_kl": 0.08,
        "aggregate_ci95": (0.06, 0.10),
        "median_delta_kl": 0.0,
        "n_prompts": 20,
        "n_prompts_significant_positive": 14,
        "n_prompts_significant_negative": 1,
        "sign_test_p_value": 0.0,
        "blocks_per_configuration": {"N4_V6": 100, "N6_V6": 100},
        "total_marginals": 61120,
    }
    values.update(overrides)
    return decide_multiprompt_gate(**values)


def test_decide_multiprompt_gate_is_conditional_when_median_is_zero():
    decision = _decide()
    assert decision["status"] == "GO conditionnel"
    assert decision["criteria"]["1_aggregate_delta_kl_positive_ci_excludes_zero"]["met"]
    assert not decision["criteria"]["2_median_delta_kl_positive"]["met"]
    criterion3 = decision["criteria"][
        "3_proportion_prompts_significant_positive_ge_threshold"
    ]
    assert criterion3["met"]
    assert criterion3["value"] == pytest.approx(0.7)
    assert criterion3["historical_ci_excludes_zero"]["n_prompts"] == 15
    assert criterion3["historical_ci_excludes_zero"][
        "n_prompts_significant_negative"
    ] == 1
    assert "1 significativement négatif" in decision["rationale"]


def test_decide_multiprompt_gate_rejects_primary_failure():
    decision = _decide(
        mean_delta_kl=-0.01,
        aggregate_ci95=(-0.02, 0.01),
        median_delta_kl=-0.1,
    )
    assert decision["status"] == "NO-GO"


def test_decide_multiprompt_gate_is_go_when_all_criteria_hold():
    decision = _decide(median_delta_kl=0.001, n_prompts_significant_negative=0)
    assert decision["status"] == "GO"


def test_decide_multiprompt_gate_uses_positive_proportion_not_absolute_significance():
    """15/20 « IC exclut zéro » ne suffit pas si un des 15 est négatif."""
    decision = _decide(
        n_prompts_significant_positive=9,
        n_prompts_significant_negative=6,
        median_delta_kl=0.001,
    )
    criterion3 = decision["criteria"][
        "3_proportion_prompts_significant_positive_ge_threshold"
    ]
    assert not criterion3["met"]
    assert criterion3["historical_ci_excludes_zero"]["n_prompts"] == 15
    assert decision["status"] == "GO conditionnel"
