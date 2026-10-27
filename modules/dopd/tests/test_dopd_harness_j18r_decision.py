"""Tests purs (sans `torch`) de la sémantique J18R — liste figée §8 du PRE.

T1 identité / non-dégénérescence, T2 mutation F4, T3 mutation F3, T4 équivalence
ligne-à-ligne inline ↔ paquet, T5 branches d'échec et issues §4.5, T6 float32
réellement appliquée, T7 séparation tests purs / torch.

Aucun `import torch` ici : le harnais est chargé par chemin (il n'importe que NumPy
et le paquet) et le fixture de scoring est un fake duck-typed. Les tests qui
nécessitent torch (T8) vivent dans `test_dopd_harness_j18r_torch.py`.
"""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from academiia.dopd.diagnostic import bootstrap_delta_ci
from academiia.dopd.target import REASON_APPLIED, compute_future_aware_target

_HARNESS_PATH = Path(__file__).resolve().parents[1] / "harness" / "run_j18r_decision.py"
_SPEC = importlib.util.spec_from_file_location("dopd_harness_j18r_decision", _HARNESS_PATH)
assert _SPEC is not None and _SPEC.loader is not None
j18r = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = j18r
_SPEC.loader.exec_module(j18r)

VOCAB = 12


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    shift = float(np.max(logits))
    exponentials = np.exp(logits - shift)
    return (logits - shift) - np.log(exponentials.sum())


def fixture_table(dominance: float = 8.0):
    """Table de log-probabilités déterministe : prior peu piqué, continuation piquée.

    Le prior causal reste informatif mais pas dégénéré (ΔKL non identitaire), tandis
    que la vraisemblance de suffixe concentre la cible sur le bloc observé — d'où le
    contrôle négatif §1.5 au résidu numérique.
    """

    def table(prefix: tuple[int, ...]) -> np.ndarray:
        if not prefix:
            logits = np.full(VOCAB, -1.0, dtype=np.float64)
            for index in range(3):
                logits[(7 * index + 5) % VOCAB] += 0.6
            return _log_softmax(logits)
        dominant = (7 * sum(prefix) + 3) % VOCAB
        logits = np.full(VOCAB, -dominance, dtype=np.float64)
        logits[dominant] = 0.0
        return _log_softmax(logits)

    return table


class FixtureScorer:
    """Professeur duck-typed, cohérent avec la table du support (Bayes-consistant)."""

    def __init__(self, table) -> None:
        self.table = table
        self.calls = 0

    def _path_logprob(self, prefix: tuple[int, ...], tokens: list[int]) -> float:
        total = 0.0
        path = tuple(prefix)
        for token in tokens:
            total += float(self.table(path)[int(token)])
            path = path + (int(token),)
        return total

    def job(self, prefix, position, tokens, suffix, reference_token):
        self.calls += 1
        prior = np.exp(self.table(tuple(prefix)))
        scores = np.array(
            [
                self._path_logprob(prefix, [int(token)]) + self._path_logprob(prefix + (int(token),), list(suffix))
                for token in tokens
            ]
        )
        reference = self._path_logprob(prefix, [int(reference_token)]) + self._path_logprob(
            prefix + (int(reference_token),), list(suffix)
        )
        return prior, scores, float(reference)

    def replay_job(self, prefix, candidates, suffix):
        prior = np.exp(self.table(tuple(prefix)))
        scores = np.array(
            [
                self._path_logprob(prefix, [int(token)]) + self._path_logprob(prefix + (int(token),), list(suffix))
                for token in candidates
            ]
        )
        return prior[list(candidates)], scores


def _run_arm(arm: str, block_size: int, vocab_width: int, blocks: int, table, integrity=None, k=None) -> list:
    support = j18r.build_economical_support(
        lambda prefixes: np.stack([table(prefix) for prefix in prefixes]), block_size, vocab_width, option="A"
    )
    sequences, log_joint = support.enumerate_joint()
    return j18r.process_configuration(
        arm=arm,
        prompt="p01",
        config_index=f"N{block_size}_V{vocab_width}",
        support=support,
        sequences=sequences,
        log_joint=log_joint,
        provider=FixtureScorer(table),
        seed=j18r.SEED_DECISION,
        blocks=blocks,
        integrity=integrity if integrity is not None else j18r.IntegrityStats(),
        replay_stats=j18r.ReplayStats(),
        block_sample=list(range(blocks)),
        future_is_state=(arm == "control"),
        k=k,
    )


def _assert_non_degenerate(observations: list) -> None:
    """Assertions T1(a) : médiane `future_kl` > 0 et ΔKL non identitaire."""
    values = [float(o.future_kl) for o in observations if o.visible_future]
    deltas = [float(o.delta_kl) for o in observations if o.visible_future]
    assert values and j18r._median(values) > 0.0
    assert any(abs(delta) > 1e-9 for delta in deltas)
    assert j18r._median(deltas) > 0.0


# ---------------------------------------------------------------------------
# T1 — identité / non-dégénérescence
# ---------------------------------------------------------------------------


def test_t1_decision_geometry_is_not_degenerate() -> None:
    """(a) futur tiré séparément : médiane `future_kl` > 0, ΔKL non identitaire."""
    observations = _run_arm("decision", 4, 6, blocks=10, table=fixture_table())
    _assert_non_degenerate(observations)


def test_t1_single_mask_control_geometry_collapses_to_zero() -> None:
    """(b) géométrie un-masque : `max(future_kl) ≤ 1e-3` et médiane ≤ 1e-5 (§1.5)."""
    observations = _run_arm("control", 4, VOCAB, blocks=3, table=fixture_table())
    values = [float(o.future_kl) for o in observations]
    assert len(values) >= 3 and all(o.arm == "control" for o in observations)
    metrics = j18r.control_metrics([observation.record() for observation in observations])
    assert max(values) <= j18r.CONTROL_MAX_FUTURE_KL
    assert j18r._median(values) <= j18r.CONTROL_MEDIAN_FUTURE_KL
    assert metrics["n"] == len(values)
    assert metrics["max_future_kl"] <= j18r.CONTROL_MAX_FUTURE_KL
    assert metrics["median_future_kl"] <= j18r.CONTROL_MEDIAN_FUTURE_KL


# ---------------------------------------------------------------------------
# T2 — mutation F4 (`future := causal`) : la suite doit échouer
# ---------------------------------------------------------------------------


def test_t2_replacing_target_by_causal_prior_makes_delta_kl_identically_zero() -> None:
    """ΔKL ≡ 0 sous F4 : l'assertion de non-dégénérescence de T1(a) tombe."""
    observations = _run_arm("decision", 4, 6, blocks=10, table=fixture_table())
    mutated = [
        j18r.MarginalObservation(
            arm=observation.arm,
            prompt=observation.prompt,
            config_index=observation.config_index,
            block_index=observation.block_index,
            position=observation.position,
            noise=observation.noise,
            visible_future=observation.visible_future,
            causal_kl=observation.causal_kl,
            future_kl=observation.causal_kl,  # F4 : T1/T2 := T0
            delta_kl=0.0,
        )
        for observation in observations
    ]
    assert all(abs(float(o.delta_kl)) == 0.0 for o in mutated)
    with pytest.raises(AssertionError):
        _assert_non_degenerate(mutated)


# ---------------------------------------------------------------------------
# T3 — mutation F3 (bootstrap par marginaux au lieu de blocs de base)
# ---------------------------------------------------------------------------


def test_t3_block_bootstrap_differs_from_marginal_bootstrap() -> None:
    """Fixture à corrélation intra-bloc : l'unité de rééchantillonnage est le bloc."""
    records = []
    for config in (0, 1):
        for block_index in range(6):
            # forte corrélation intra-bloc : les marginaux d'un bloc portent la même valeur
            value = 0.05 + 0.35 * ((block_index * 7 + config) % 5) / 5.0
            for position in range(4):
                records.append(
                    {
                        "prompt": "p01",
                        "config_index": config,
                        "block_index": block_index,
                        "position": position,
                        "causal_kl": value + 0.02,
                        "future_kl": value,
                        "delta_kl": 0.02,
                    }
                )
    assert j18r.GROUP_FIELDS == ("prompt", "config_index", "block_index")
    by_block = bootstrap_delta_ci(records, 20261028, 4000, group_fields=j18r.GROUP_FIELDS)
    by_marginal = bootstrap_delta_ci(records, 20261028, 4000, group_fields=("prompt", "config_index", "position"))
    assert by_block[0] != by_marginal[0] or by_block[1] != by_marginal[1]
    # la clé de regroupement est bien le bloc de base, pas le marginal
    keys = {tuple(record[field] for field in j18r.GROUP_FIELDS) for record in records}
    assert len(keys) == 2 * 6
    assert by_block[0] < by_marginal[0] or by_block[1] > by_marginal[1]


# ---------------------------------------------------------------------------
# T4 — équivalence ligne-à-ligne inline (T1 complet) ↔ paquet eq. 18 tronquée
# ---------------------------------------------------------------------------


def test_t4_inline_full_target_matches_package_on_full_support_in_finite_regime() -> None:
    """Deltas, facteurs et normaliseur au relatif ≤ 1e-6 ; cible finale ≤ 1e-5."""
    table = fixture_table(dominance=2.0)
    prefix = (3, 5)
    prior = np.exp(table(prefix))
    tokens = list(range(VOCAB))
    suffix = [7, 1]
    scorer = FixtureScorer(table)
    _, scores, reference = scorer.job(prefix, 1, tokens, suffix, reference_token=4)
    inline = j18r.inline_full_target(prior, scores, reference)
    assert inline["applied"] and inline["reason"] == REASON_APPLIED

    package = compute_future_aware_target(
        prior,
        candidate_indices=tokens,
        candidate_loglikelihoods=list(scores),
        reference_loglikelihood=reference,
        visible_future=True,
        arithmetic_dtype="float32",
    )
    assert package.applied and package.reason == REASON_APPLIED
    assert package.n_candidates == VOCAB

    for index, value in enumerate(inline["deltas"]):
        expected = np.float32(scores[index]) - np.float32(reference)
        assert abs(float(value) - float(expected)) <= 1e-6 * max(1.0, abs(float(expected)))
    for index, value in enumerate(inline["factors"]):
        expected = float(np.exp(np.float32(inline["deltas"][index])))
        assert abs(float(value) - expected) <= 1e-6 * max(1.0, expected)
    manual_weights = np.asarray(prior, dtype=np.float32) * np.exp(np.asarray(inline["deltas"], dtype=np.float32))
    manual_normalizer = float(np.sum(manual_weights, dtype=np.float32))
    assert abs(float(inline["normalizer"]) - manual_normalizer) <= 1e-6 * abs(manual_normalizer)
    assert float(np.max(np.abs(np.asarray(inline["target"], dtype=np.float64) - np.asarray(package.target, dtype=np.float64)))) <= 1e-5


# ---------------------------------------------------------------------------
# T5 — branches d'échec C4–C10 et issues de la table §4.5
# ---------------------------------------------------------------------------


def _criteria(**overrides: bool) -> dict[str, bool]:
    flags = {name: True for name in j18r.CRITERIA_KEYS}
    flags.update({"C1": True, "C1_sigma_positive": True, "arms_complete": True})
    flags.update(overrides)
    return flags


def test_t5_all_criteria_hold_gives_go() -> None:
    decision = j18r.decide_j18r_issue(_criteria(), {"R_a": False, "R_b": False, "R_c": False})
    assert decision["status"] == "GO" and decision["priority"] == 3


def test_t5_reserve_or_sigma_failure_gives_conditional_go() -> None:
    decision = j18r.decide_j18r_issue(_criteria(C1_sigma_positive=False), {"R_a": False, "R_b": True, "R_c": False})
    assert decision["status"] == "GO conditionnel" and decision["priority"] == 4
    assert set(decision["failed_criteria"]) == {"C1_sigma_positive", "R_b"}


def test_t5_c1_failure_on_valid_run_gives_no_go() -> None:
    """ΔKL ≤ 0 ou IC ∋ 0 sur run valide : NO-GO, pas INVALIDE."""
    decision = j18r.decide_j18r_issue(_criteria(C1=False), {"R_a": False, "R_b": False, "R_c": False})
    assert decision["status"] == "NO-GO" and decision["priority"] == 5


def test_t5_anchor_failure_beats_every_validity_blocker_with_stop() -> None:
    """Ancrage hors clause C9(a) ou C9(b) : STOP, priorité 1, même si C4 et C6 échouent."""
    decision = j18r.decide_j18r_issue(_criteria(C9=False, C4=False, C6=False), {"R_a": False, "R_b": False, "R_c": False})
    assert decision["status"] == "STOP" and decision["priority"] == 1


@pytest.mark.parametrize("name", j18r.CRITERIA_KEYS)
def test_t5_each_validity_branch_gives_invalide(name: str) -> None:
    """Rejeu > 1e-3 (C5), hashes incomplets (C6), professeur non gelé (C7), modèle hors
    dépôt (C8), effectif < 2000 marginaux (C4), hash `target.py` modifié (C10)."""
    if name == "C9":
        pytest.skip("C9 est l'ancrage : il donne STOP (§4.5), pas INVALIDE")
    decision = j18r.decide_j18r_issue(_criteria(**{name: False}), {"R_a": False, "R_b": False, "R_c": False})
    assert decision["status"] == "INVALIDE" and decision["priority"] == 2
    assert decision["failed_criteria"] == [name]


def test_t5_incomplete_required_arm_gives_invalide_not_no_go() -> None:
    decision = j18r.decide_j18r_issue(_criteria(arms_complete=False), {"R_a": True, "R_b": False, "R_c": False})
    assert decision["status"] == "INVALIDE"
    assert "bras requis incomplet" in decision["failed_criteria"]


def test_t5_reserves_are_evaluated_on_decision_publications() -> None:
    publications = {
        "median_delta_kl_visible_future": -0.01,
        "degraded_fraction": 0.6,
        "per_noise": {"1.0": {"mean_delta_kl": -0.02}, "1.5": {"mean_delta_kl": 0.03}},
    }
    reserves = j18r.evaluate_reserves(publications)
    assert reserves == {"R_a": True, "R_b": True, "R_c": True}
    assert j18r.evaluate_reserves(
        {"median_delta_kl_visible_future": 0.02, "degraded_fraction": 0.3, "per_noise": {"1.0": {"mean_delta_kl": 0.05}}}
    ) == {"R_a": False, "R_b": False, "R_c": False}


# ---------------------------------------------------------------------------
# T6 — float32 réellement appliquée (écart G)
# ---------------------------------------------------------------------------


def test_t6_float32_arithmetic_reproduces_float32_reference_and_differs_from_float64() -> None:
    """Fixture où float32 et float64 donnent des cibles distinctes."""
    # prior et scores choisis pour que le rounding float32 soit visible sur la cible
    prior = np.asarray([0.42, 0.31, 0.17, 0.10], dtype=np.float64)
    scores = [-1000.0, -1000.3, -999.85, -1000.1]
    reference = -1000.05
    candidates = list(range(prior.size))

    float32_result = compute_future_aware_target(
        prior, candidate_indices=candidates, candidate_loglikelihoods=scores, reference_loglikelihood=reference,
        visible_future=True, arithmetic_dtype="float32",
    )
    float64_result = compute_future_aware_target(
        prior, candidate_indices=candidates, candidate_loglikelihoods=scores, reference_loglikelihood=reference,
        visible_future=True, arithmetic_dtype="float64",
    )
    assert float32_result.applied and float64_result.applied
    reference_target = j18r.float32_reference_target(prior, candidates, scores, reference)
    assert np.max(np.abs(np.asarray(float32_result.target, dtype=np.float64) - reference_target)) <= 1e-9

    distinction = float(np.max(np.abs(np.asarray(float32_result.target, dtype=np.float64) - np.asarray(float64_result.target, dtype=np.float64))))
    assert distinction > 1e-7


def test_t6_harness_records_float32_application_on_every_decision_marginal() -> None:
    """La voie de production applique bien l'arithmetique float32 (cartographie G)."""
    integrity = j18r.IntegrityStats()
    observations = _run_arm("decision", 4, 6, blocks=4, table=fixture_table(), integrity=integrity, k=4)
    records = [observation.record() for observation in observations]
    required = set(j18r.RECORD_FIELDS)
    assert len(records) == len(observations) and all(required <= set(record) for record in records)
    applied = sum(1 for observation in observations if observation.reason == "applied")
    assert applied > 0 and integrity.float32_targets_seen == applied and integrity.float32_applied
    assert integrity.float32_mismatches == 0
    # garde : une cible calculee en float64 mais declaree float32 est signalee
    prior = np.asarray([0.42, 0.31, 0.17, 0.10], dtype=np.float64)
    scores = [-1000.0, -1000.3, -999.85, -1000.1]
    reference = -1000.05
    guarded = j18r.IntegrityStats()
    float64_target = compute_future_aware_target(
        prior,
        candidate_indices=list(range(prior.size)),
        candidate_loglikelihoods=scores,
        reference_loglikelihood=reference,
        visible_future=True,
        arithmetic_dtype="float64",
    ).target
    guarded.observe_float32_application("float32", True, float64_target, j18r.float32_reference_target(prior, list(range(prior.size)), scores, reference))
    assert guarded.float32_targets_seen == 1 and guarded.float32_mismatches == 1


# ---------------------------------------------------------------------------
# T7 — séparation tests purs / torch
# ---------------------------------------------------------------------------


def test_t7_pure_test_file_never_imports_torch() -> None:
    source = Path(__file__).resolve().read_text(encoding="utf-8")
    assert not re.search(r"^\s*import torch|^\s*from torch", source, flags=re.MULTILINE)
    harness_source = _HARNESS_PATH.read_text(encoding="utf-8")
    assert not re.search(r"^import torch|^from torch", harness_source, flags=re.MULTILINE)


def test_t7_harness_loads_and_collects_without_torch_in_a_fresh_process() -> None:
    """La voie NumPy se charge et se collecte sans `torch` (convention J15 P3)."""
    code = (
        "import importlib.util, json, sys\n"
        "class Blocker:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name == 'torch' or name.startswith('torch.'):\n"
        "            raise ImportError('torch est bloque pour ce controle')\n"
        "        return None\n"
        "sys.meta_path.insert(0, Blocker())\n"
        f"_SPEC = importlib.util.spec_from_file_location('h', {str(_HARNESS_PATH)!r})\n"
        "module = importlib.util.module_from_spec(_SPEC)\n"
        "sys.modules['h'] = module\n"
        "_SPEC.loader.exec_module(module)\n"
        "try:\n"
        "    import torch\n"
        "    imported = True\n"
        "except ImportError:\n"
        "    imported = False\n"
        "import pytest\n"
        f"collected = pytest.main(['--collect-only', '-q', {str(Path(__file__))!r}])\n"
        "print(json.dumps({'torch_imported': imported, 'torch_absent': 'torch' not in sys.modules, "
        "'collection_ok': collected == 0, 'control_ok': module.control_metrics([]) is not None}))\n"
    )
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout.splitlines()[-1])
    assert not payload["torch_imported"] and payload["torch_absent"]
    assert payload["collection_ok"] and payload["control_ok"]


def test_t7_torch_tests_are_isolated_in_a_separate_file() -> None:
    torch_file = Path(__file__).resolve().parent / "test_dopd_harness_j18r_torch.py"
    assert torch_file.is_file()
    source = torch_file.read_text(encoding="utf-8")
    assert re.search(r"^torch = pytest\.importorskip\(", source, flags=re.MULTILINE)

