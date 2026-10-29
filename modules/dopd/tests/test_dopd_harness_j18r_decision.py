"""Tests purs (sans `torch`) de la sémantique J18R — liste figée §8 du PRE.

T1 identité / non-dégénérescence, T2 mutation F4, T3 mutation F3, T4 équivalence
ligne-à-ligne inline ↔ paquet, T5 branches d'échec et issues §4.5, T6 float32
réellement appliquée, T7 séparation tests purs / torch, R-J18R-01…04 corrections bloquantes
(revue N1 sur `c42dab7`), R-J18R-05 gate T1 + troncature, R-J18R-06 hash §9.2, R-J18R-07…12 corrections
bloquantes de la revue N2 (ancrage, bruts, invocation canonique, C9 non mesurable,
préfixe conditionné sur le prompt).

Aucun `import torch` ici : le harnais est chargé par chemin (il n'importe que NumPy
et le paquet) et le fixture de scoring est un fake duck-typed. Les tests qui
nécessitent torch (T8) vivent dans `test_dopd_harness_j18r_torch.py`.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import re
import shlex
import shutil
import subprocess
import sys
import types
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest

from academiia.core import sha256_file
from academiia.dopd.diagnostic import bootstrap_delta_ci
from academiia.dopd.target import REASON_APPLIED, REASON_NO_VISIBLE_FUTURE, compute_future_aware_target

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


def _run_arm(
    arm: str,
    block_size: int,
    vocab_width: int,
    blocks: int,
    table,
    integrity=None,
    k=None,
    module=None,
) -> list:
    """Exécute un bras sur la fixture ; `module` permet d'injecter un harnais muté."""
    mod = module if module is not None else j18r
    support = mod.build_economical_support(
        lambda prefixes: np.stack([table(prefix) for prefix in prefixes]), block_size, vocab_width, option="A"
    )
    sequences, log_joint = support.enumerate_joint()
    return mod.process_configuration(
        arm=arm,
        prompt="p01",
        config_index=f"N{block_size}_V{vocab_width}",
        support=support,
        sequences=sequences,
        log_joint=log_joint,
        provider=FixtureScorer(table),
        seed=mod.SEED_DECISION,
        blocks=blocks,
        integrity=integrity if integrity is not None else mod.IntegrityStats(),
        replay_stats=mod.ReplayStats(),
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


def _reserves(per_noise: dict[str, dict[str, object]]) -> dict[str, bool]:
    """R_a/R_b fixés, R_c variable : isole la réserve §4.2 du reste."""
    return j18r.evaluate_reserves(
        {"median_delta_kl_visible_future": -0.01, "degraded_fraction": 0.6, "per_noise": per_noise}
    )


def test_t5_reserves_a_and_b_use_median_and_degraded_fraction() -> None:
    reserves = _reserves({"1.0": {"mean_delta_kl": 0.03}, "1.5": {"mean_delta_kl": 0.03}})
    assert reserves["R_a"] is True and reserves["R_b"] is True
    assert j18r.evaluate_reserves({"median_delta_kl_visible_future": 0.02, "degraded_fraction": 0.3}) == {
        "R_a": False,
        "R_b": False,
        "R_c": False,
    }


# ---------------------------------------------------------------------------
# R-J18R-01 — réserve R_c : quantificateur universel sur les niveaux attendus (§4.2)
# ---------------------------------------------------------------------------


def test_r_j18r_01_reserve_c_is_universal_not_existential() -> None:
    """PRE §4.2 « pour tout σ ≥ 1,0 » : un seul niveau positif empêche la réserve."""
    assert _reserves({"1.0": {"mean_delta_kl": -0.02}, "1.5": {"mean_delta_kl": 0.03}})["R_c"] is False


def test_r_j18r_01_reserve_c_holds_when_all_expected_levels_are_non_positive() -> None:
    assert _reserves({"1.0": {"mean_delta_kl": -0.02}, "1.5": {"mean_delta_kl": 0.0}})["R_c"] is True


def test_r_j18r_01_reserve_c_does_not_hold_without_non_positive_levels() -> None:
    assert _reserves({"1.0": {"mean_delta_kl": 0.05}, "1.5": {"mean_delta_kl": 0.02}})["R_c"] is False


def test_r_j18r_01_reserve_c_stays_untriggered_on_incomplete_data() -> None:
    """Niveau attendu absent (grille figée §1.2) : pas de réserve sur données partielles."""
    assert _reserves({"1.0": {"mean_delta_kl": -0.02}})["R_c"] is False
    assert _reserves({})["R_c"] is False


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
    # `--capture=sys` dans la sous-invocation : la capture par défaut crée une
    # tmpfile sous `TMPDIR`, et DrvFs (`/mnt/d`) ne supporte pas `O_TMPFILE` —
    # la sous-collection échouerait en `FileNotFoundError` au truncate de fin de
    # session quand `TMPDIR` est redirigé dans le dépôt (règle 9).
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
        f"collected = pytest.main(['--collect-only', '-q', '--capture=sys', {str(Path(__file__))!r}])\n"
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

# ---------------------------------------------------------------------------
# R-J18R-02 — C1 : critère agrégé ; les niveaux par σ restent des publications (§4.2)
# ---------------------------------------------------------------------------


def _row(arm: str, prompt: str, config_index: int, block_index: int, noise: float, delta: float) -> dict:
    """Observation brute sur les champs de `RECORD_FIELDS` (`delta_kl = causal - future`)."""
    return {
        "arm": arm,
        "prompt": prompt,
        "config_index": config_index,
        "block_index": block_index,
        "position": block_index * 4,
        "noise": float(noise),
        "visible_future": True,
        "causal_kl": float(delta),
        "future_kl": 0.0,
        "delta_kl": float(delta),
        "full_kl": float(delta) + 0.02,
        "prod_kl_teacheron": float(delta) + 0.03,
        "delta_kl_teacheron": float(delta) + 0.03,
        "n_candidates": 4,
        "mass_teacher": 1.0,
        "mass_student": 0.95,
    }


def test_r_j18r_02_c1_holds_on_the_aggregate_when_one_noise_level_fails() -> None:
    """PRE §4.2 : C1 = agrégat > 0 et IC excluant zéro ; un σ non significatif ne le durcit pas."""
    records = [
        *[_row("decision", f"p{index:02d}", 0, index, 0.0, 0.10) for index in range(40)],
        *[_row("decision", f"p{index:02d}", 0, index, 1.0, -0.02) for index in range(40)],
    ]
    publications = j18r.arm_publications(
        j18r.observations_from_records(records), arm="decision", bootstrap_seed=11, iterations=300
    )
    c1 = j18r.criterion_c1(records, publications, bootstrap_seed=11, iterations=300)
    # chaque bloc de base (prompt, config, bloc) est positif : l'IC agrégé exclut zéro
    assert c1["aggregate"]["ci95"][0] > 0.0
    failing = sorted(key for key, entry in c1["per_noise"].items() if not entry["holds"])
    assert failing == ["1.0"]
    assert c1["holds"] is True
    # la publication par σ et le critère distinct `C1_sigma_positive` restent publiés
    assert publications.get("c1_by_noise", {}).get("sigma_positive_only")


# ---------------------------------------------------------------------------
# R-J18R-03 — C4 : effectif du contrôle = échantillon figé §1.5, pas la liste des prompts
# ---------------------------------------------------------------------------


def _control_records() -> list[dict]:
    return [
        _row("control", prompt, config_index, block_index, 0.0, 0.001)
        for config_index in range(len(j18r.CONTROL_CONFIGS))
        for prompt in j18r.CONTROL_PROMPTS
        for block_index in j18r.CONTROL_BLOCKS
    ]


def _decision_records(n_prompts: int = 20, blocks: int = 100) -> list[dict]:
    return [
        _row("decision", f"p{prompt:02d}", config_index, block_index, 0.0, 0.05)
        for config_index in range(len(j18r.DECISION_CONFIGS))
        for prompt in range(n_prompts)
        for block_index in range(blocks)
    ]


def test_r_j18r_03_control_effectif_uses_the_frozen_sample_not_the_prompt_list() -> None:
    """Une liste complète de 20 prompts ne doit pas faire attendre 60 blocs au contrôle."""
    publications = {
        "control": j18r.arm_publications(
            j18r.observations_from_records(_control_records()), arm="control", bootstrap_seed=5, iterations=25
        ),
        "decision": j18r.arm_publications(
            j18r.observations_from_records(_decision_records()), arm="decision", bootstrap_seed=5, iterations=25
        ),
    }
    c4 = j18r.criterion_c4(publications, blocks=100, n_prompts=20)
    assert c4["holds"] is True
    control_entries = [entry for key, entry in c4["per_config"].items() if key.startswith("control/")]
    decision_entries = [entry for key, entry in c4["per_config"].items() if key.startswith("decision/")]
    assert len(control_entries) == 2 and len(decision_entries) == 2
    assert all(
        entry["expected_blocks_sampled"] == len(j18r.CONTROL_PROMPTS) * len(j18r.CONTROL_BLOCKS)
        for entry in control_entries
    )
    assert all(entry["expected_blocks_sampled"] == 20 * 100 for entry in decision_entries)


def test_r_j18r_03_control_c4_still_fails_on_an_incomplete_sample() -> None:
    """C4 n'est pas devenu permissif : un bloc de contrôle manquant fait échouer le critère."""
    incomplete = [row for row in _control_records() if not (row["prompt"] == "p03" and row["block_index"] == 2)]
    publications = {
        "control": j18r.arm_publications(
            j18r.observations_from_records(incomplete), arm="control", bootstrap_seed=5, iterations=25
        )
    }
    c4 = j18r.criterion_c4(publications, blocks=100, n_prompts=20)
    assert c4["holds"] is False
    entry = c4["per_config"]["control/0"]
    assert entry["n_blocks_sampled"] == 8 and entry["expected_blocks_sampled"] == 9


# ---------------------------------------------------------------------------
# R-J18R-04 — N6 conditionné par `--n6-chat-template`, dans le plan et dans l'exécution (§6)
# ---------------------------------------------------------------------------


class _FakeTorchModule:
    __version__ = "2.12.0-fake"

    class cuda:
        @staticmethod
        def is_available() -> bool:
            return False


class _FakeTransformersModule:
    __version__ = "5.10.2-fake"

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(name):
            return "fake-tokenizer"


class _FakeModel:
    training = False

    def parameters(self) -> list:
        return []


_MAX_PREFIX_CACHE_BYTES = 16 * 1024**3


class _MirrorScorer:
    """Miroir exact de la signature de production (`teacher_scorer_qwen3.py:253-259`) :
    `Qwen3TeacherScorer(model, *, device=…, max_prefix_cache_bytes=…)` ; le tokenizer n'y
    figure pas — il est hors du contrat `TeacherScorer` (D1 du constat §9.5)."""

    def __init__(self, model, *, device="cpu", max_prefix_cache_bytes=_MAX_PREFIX_CACHE_BYTES) -> None:
        self.model = model
        self.device = device
        self.max_prefix_cache_bytes = max_prefix_cache_bytes


def _make_scorer_module(recorder: list[str]):
    """Module de scoring simulé portant la signature réelle ; enregistre le `device` reçu."""

    class MirrorScorer(_MirrorScorer):
        def __init__(self, model, **kwargs) -> None:
            super().__init__(model, **kwargs)
            recorder.append(str(self.device))

    def load_teacher(model, device=None, dtype=None):
        return _FakeModel(), "fake-source"

    return types.SimpleNamespace(Qwen3TeacherScorer=MirrorScorer, load_teacher=load_teacher)


def _fake_execute_environment(monkeypatch, *, returncode: int = 0, write_raw: bool = True) -> dict[str, object]:
    """Exécution sans torch ni GPU : seules les collaborations externes sont simulées.

    La sous-invocation J15 écrit son shard brut sous le `--output` de son invocation, comme
    le harnais gelé §7.3b ; `ingest_arm_shard`, `write_json` et le sidecar restent **réels**
    (R-J18R-11 : le chemin complet de `execute` est exercé, pas sa seule sélection d'arms).
    """
    arms: list[str] = []
    scorer_devices: list[str] = []
    raw_payloads: dict[str, str] = {}
    monkeypatch.setattr(j18r, "_torch", lambda: _FakeTorchModule)
    # `execute` charge le module de scoring **réel** (torch) et `transformers` : les deux
    # sont simulés ici, sinon la suite pure exigerait torch (absent en CI).
    monkeypatch.setattr(j18r, "_transformers", lambda: _FakeTransformersModule)
    monkeypatch.setattr(j18r, "load_scorer_module", lambda: _make_scorer_module(scorer_devices))
    # Le garde-fou §7.7 résout un snapshot HF réel : hors CI (pas de cache), refus sortie 2.
    monkeypatch.setattr(
        j18r,
        "verify_model_source_in_repo",
        lambda model, root: {"model_source_in_repo": True, "model_snapshot": "fake", "model_revision": "fake"},
    )

    def raw_rows(arm: str) -> list[dict]:
        rows = [_row(arm, f"p{p + 1:02d}", "N4_V6", b, 0.0, 0.05) for p in range(2) for b in range(3)]
        for row in rows:
            row.pop("arm")  # J15 n'étiquette pas : J18R étiquette à l'ingestion
        return rows

    def fake_subprocess(invocation, root):
        tokens = shlex.split(invocation)
        output = root / Path(tokens[tokens.index("--output") + 1])
        arm = {
            j18r.RAW_SHARDS["anchor"]: "anchor",
            j18r.RAW_SHARDS["sensitivity_fp32"]: "sensitivity_fp32",
        }[output.name]
        if write_raw:
            payload = json.dumps({"rows": raw_rows(arm)}, indent=2) + "\n"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(payload, encoding="utf-8")
            raw_payloads[str(output.relative_to(root))] = payload
        return returncode

    monkeypatch.setattr(j18r, "run_arm_subprocess", fake_subprocess)

    def fake_records(arm: str) -> list[dict]:
        return [
            _row(arm, f"p{prompt:02d}", "N4_V6", block_index, 0.0, 0.05)
            for prompt in range(2)
            for block_index in range(3)
        ]

    def publications_for(arm: str, records: list[dict]) -> dict[str, object]:
        return j18r.arm_publications(
            j18r.observations_from_records(records), arm=arm, bootstrap_seed=5, iterations=25
        )

    def fake_run_arm(**kwargs):
        arm = str(kwargs["arm"])
        arms.append(arm)
        # le contrat de `run_arm` est un fabricant de provider par prompt (R-J18R-12) ;
        # le provider porte le `device` lié par mot-cle (R-J18R-07).
        provider = kwargs["provider_factory"]((7, 8, 9))
        assert isinstance(provider, j18r.ProductionScoreProvider)
        # le `device` est lié par mot-clé au constructeur : le miroir l'enregistre (§R-J18R-07)
        assert str(provider._scorer.device) == str(kwargs["args"].device)
        records = fake_records(arm)
        return {"arm": arm, "records": records, "publications": publications_for(arm, records), "complete": True}

    # `run_arm_guarded` (R-J18R-14) reste **réel** : c'est lui qui intercepte les
    # exceptions du seul bras N6 ; `run_arm` est simulé en dessous.
    monkeypatch.setattr(j18r, "run_arm", fake_run_arm)
    return {"arms": arms, "scorer_devices": scorer_devices, "raw_payloads": raw_payloads}


def _scratch_output_dir(name: str) -> Path:
    """Sortie de test dans `_artefact/` (règle 9) ; `ArtifactRef` exige un chemin relatif au dépôt."""
    out = Path("_artefact") / "j18r_decision_tests" / name
    resolved = j18r._REPO_ROOT / out
    if resolved.exists():
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)
    return out


def test_r_j18r_04_execute_skips_n6_when_the_flag_is_absent(monkeypatch) -> None:
    env = _fake_execute_environment(monkeypatch)
    out = _scratch_output_dir("n6_absent")
    argv = ["--output-dir", str(out), "--blocks", "2", "--bootstrap-iterations", "25"]
    args = j18r.build_parser().parse_args(argv)
    outcome = j18r.execute(args, argv)
    assert env["arms"] == ["control", "decision"]
    results = json.loads((j18r._REPO_ROOT / out / "results.json").read_text(encoding="utf-8"))
    assert "n6_chat_template" not in results["arms"]
    assert "arm_n6.json" not in outcome["shards"]
    assert not [key for key in results["criteria"]["C4"]["per_config"] if key.startswith("n6_chat_template/")]
    # le plan et l'exécution partagent la même sélection, dans le même ordre (délégués puis exécutés)
    assert list(j18r.dry_run_plan(argv)["arms"]) == ["anchor", "sensitivity_fp32", "control", "decision"]


def test_r_j18r_04_execute_runs_n6_when_the_flag_is_present(monkeypatch) -> None:
    env = _fake_execute_environment(monkeypatch)
    out = _scratch_output_dir("n6_present")
    argv = ["--output-dir", str(out), "--blocks", "2", "--bootstrap-iterations", "25", "--n6-chat-template"]
    args = j18r.build_parser().parse_args(argv)
    outcome = j18r.execute(args, argv)
    assert env["arms"] == ["control", "decision", "n6_chat_template"]
    results = json.loads((j18r._REPO_ROOT / out / "results.json").read_text(encoding="utf-8"))
    assert "n6_chat_template" in results["arms"]
    assert "arm_n6.json" in outcome["shards"]
    # R-J18R-14 : N6 est **non décisionnel** (PRE §3.2, §4.5) — il ne figure pas dans C4,
    # dont le seuil d'effectif ne concerne que les bras de mesure (ancrage, décision).
    assert not [
        key
        for key in results["criteria"]["C4"]["per_config"]
        if key.startswith("n6_chat_template/")
    ]
    assert list(j18r.dry_run_plan(argv)["arms"]) == [
        "anchor",
        "sensitivity_fp32",
        "control",
        "decision",
        "n6_chat_template",
    ]


# ---------------------------------------------------------------------------
# R-J18R-05 — gate T1 (§1.4) et publication de la troncature (§4.4, revue N2)
# ---------------------------------------------------------------------------


def _load_mutated_harness(name: str, old: str, new: str):
    """Copie mutée du harnais (mutation au niveau source) dans un module frais."""
    source = _HARNESS_PATH.read_text(encoding="utf-8")
    mutated = source.replace(old, new)
    assert mutated != source, f"le motif de mutation est absent de la source : {old!r}"
    spec = importlib.util.spec_from_file_location(name, _HARNESS_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(mutated, str(_HARNESS_PATH), "exec"), module.__dict__)
    return module


_GATE_REMOVAL = (
    "    if visible_future:\n        full_result = inline_full_target(causal, scores, reference_score)",
    "    if True:\n        full_result = inline_full_target(causal, scores, reference_score)",
)


def _decision_with_k(blocks: int = 10) -> list:
    return _run_arm("decision", 4, VOCAB, blocks=blocks, table=fixture_table(), k=4)


def test_r_j18r_05_full_target_is_gated_to_causal_without_visible_future() -> None:
    """(a) bras décision, `k` défini : sans futur visible, T1 := T0, troncature nulle."""
    observations = _decision_with_k()
    invisible = [o for o in observations if not o.visible_future]
    assert invisible, "la fixture doit produire des marginaux sans futur visible"
    for observation in invisible:
        record = observation.record()
        # T1 est la cible causale elle-même : même KL sur la même cible, égalité bit à bit.
        assert observation.full_kl == observation.causal_kl
        # la ligne publiée `full_kl - future_kl` est nulle (T2 = prior renormalisé, §2.3)
        assert abs(float(record["full_kl"]) - float(record["future_kl"])) <= 1e-12
        assert abs(observation.delta_kl) <= 1e-12
        assert observation.reason == REASON_NO_VISIBLE_FUTURE


def test_r_j18r_05_anchor_arm_applies_the_same_gate_as_j15() -> None:
    """Bras ancrage (T1 seul, sans k) : la règle J15 (`future := causal`) est respectée."""
    observations = _run_arm("anchor", 4, VOCAB, blocks=6, table=fixture_table())
    invisible = [o for o in observations if not o.visible_future]
    assert invisible
    for observation in invisible:
        assert observation.future_kl == observation.causal_kl
        assert observation.delta_kl == 0.0
        assert observation.reason == REASON_NO_VISIBLE_FUTURE


def test_r_j18r_05_gate_removal_mutation_breaks_the_full_target_property() -> None:
    """(b) retirer le gate T1 casse la propriété (a) : la suite est sensible au gate."""
    mutated = _load_mutated_harness("dopd_harness_j18r_decision_gate_removed", *_GATE_REMOVAL)
    observations = _run_arm("decision", 4, VOCAB, blocks=10, table=fixture_table(), k=4, module=mutated)
    broken = [o for o in observations if not o.visible_future and o.full_kl != o.causal_kl]
    assert broken, "sans le gate, T1 applique à tort les scores d'un futur échantillonné"
    for observation in broken:
        # sonde N2 : la troncature de ces lignes devient fausse (non nulle)
        assert abs(float(observation.full_kl) - float(observation.future_kl)) > 1e-9
    # T2 reste gated par le paquet : la dérive vient de T1 seul
    assert all(abs(o.future_kl - o.causal_kl) <= 1e-12 for o in observations if not o.visible_future)


def test_r_j18r_05_t1_and_t2_agree_with_the_causal_target_without_visible_future() -> None:
    """(c) cohérence T1/T2 sur les marginaux sans futur visible : cible causale des deux côtés."""
    observations = _decision_with_k()
    invisible = [o for o in observations if not o.visible_future]
    assert invisible
    for observation in invisible:
        assert observation.full_kl == observation.causal_kl
        assert abs(observation.future_kl - observation.causal_kl) <= 1e-12
    # le gate du paquet rend la causale renormalisée, pas une correction (ligne type)
    table = fixture_table()
    support = j18r.build_economical_support(
        lambda prefixes: np.stack([table(prefix) for prefix in prefixes]), 4, VOCAB, option="A"
    )
    causal = support.causal_prior((1, 2, 3), 3)
    result = j18r.production_target(causal, list(range(VOCAB)), np.zeros(VOCAB), 0.0, visible_future=False)
    assert result.reason == REASON_NO_VISIBLE_FUTURE
    assert not result.applied
    assert np.max(np.abs(result.target - causal / causal.sum())) <= 1e-12


def test_r_j18r_05_truncation_cost_is_published_on_all_and_on_the_visible_subset() -> None:
    """§4.4 : troncature sur tous les marginaux du bras décision ET sur le sous-ensemble visible."""
    observations = _decision_with_k()
    publications = j18r.arm_publications(
        observations, arm="decision", bootstrap_seed=j18r.BOOTSTRAP_SEED_DECISION, iterations=200
    )
    all_cost = publications["truncation_cost"]
    visible_cost = publications["truncation_cost_visible_future"]
    visible = [o for o in observations if o.visible_future]
    assert all_cost["n"] == len(observations)
    assert visible_cost["n"] == len(visible)
    visible_trunc = [float(o.full_kl) - float(o.future_kl) for o in visible]
    invisible_trunc = [float(o.full_kl) - float(o.future_kl) for o in observations if not o.visible_future]
    # la moyenne sur les lignes sans futur visible est nulle : l'agrégat « tous » n'est pas
    # porté par elles (c'est la sonde de la revue N2, moyenne fausse avant correction)
    assert abs(float(np.mean(invisible_trunc))) <= 1e-12
    assert abs(visible_cost["mean"] - float(np.mean(visible_trunc))) <= 1e-12
    assert abs(all_cost["mean"] - float(np.mean(visible_trunc + invisible_trunc))) <= 1e-12
    assert visible_cost["min"] <= visible_cost["mean"] <= visible_cost["max"]


# ---------------------------------------------------------------------------
# R-J18R-06 — hash figé §9.2 de l'artefact J18 (C10 falsifiable)
# ---------------------------------------------------------------------------


def test_r_j18r_06_frozen_hash_of_j18_decision_artifact_matches_the_file() -> None:
    """Le hash épinglé est le hash réel de `j18_decision/results.json` (constat R2)."""
    relative = "modules/dopd/eval/j18_decision/results.json"
    assert j18r.FROZEN_HASHES[relative] == sha256_file(str(j18r._REPO_ROOT / relative))
    result = j18r.criterion_c10(repo_root=j18r._REPO_ROOT)
    assert result["frozen_files"][relative] is True
    assert result["holds"] is True


# ---------------------------------------------------------------------------
# R-J18R-07 — constructeur du scorer de production (`model` + `device`, pas de tokenizer)
# ---------------------------------------------------------------------------


def test_r_j18r_07_scorer_signature_is_model_plus_device_keyword() -> None:
    """D1 : le constructeur de production n'accepte pas de tokenizer positionnel ; le harnais
    doit lier `device` par mot-cle (`teacher_scorer_qwen3.py:253-259`)."""
    signature = inspect.signature(_MirrorScorer)
    bound = signature.bind(_FakeModel(), device="cuda")
    bound.apply_defaults()
    assert bound.arguments["device"] == "cuda"
    with pytest.raises(TypeError):
        signature.bind(_FakeModel(), "fake-tokenizer")


def test_r_j18r_07_execute_passes_the_device_keyword_to_the_scorer(monkeypatch) -> None:
    env = _fake_execute_environment(monkeypatch)
    out = _scratch_output_dir("scorer_device")
    argv = ["--output-dir", str(out), "--blocks", "2", "--bootstrap-iterations", "25", "--device", "cpu"]
    j18r.execute(j18r.build_parser().parse_args(argv), argv)
    # un provider par bras exécuté (control, decision), chacun lié au `device` demandé
    assert env["scorer_devices"] == ["cpu", "cpu"]
    assert env["arms"] == ["control", "decision"]


# ---------------------------------------------------------------------------
# R-J18R-08 — shard brut ingéré = fichier écrit par J15, jamais réécrit par J18R
# ---------------------------------------------------------------------------


def test_r_j18r_08_raw_shards_stay_untouched_and_are_hashed_before_ingestion(monkeypatch) -> None:
    """D2 : `ARM_SHARDS["sensitivity_fp32"]` est le nom du brut ; le `.replace(".json",
    "_raw.json")` d'ingestion inventait `arm_sensitivity_raw_raw.json`, jamais produit."""
    env = _fake_execute_environment(monkeypatch)
    out = _scratch_output_dir("raw_intact")
    argv = ["--output-dir", str(out), "--blocks", "2", "--bootstrap-iterations", "25"]
    outcome = j18r.execute(j18r.build_parser().parse_args(argv), argv)
    base = j18r._REPO_ROOT / out
    for relative in ("arm_anchor_raw.json", "arm_sensitivity_raw.json"):
        path = base / relative
        # le brut ingéré reste l'octet écrit par la sous-invocation J15 : J18R ne le réécrit pas
        assert path.read_text(encoding="utf-8") == env["raw_payloads"][f"{out}/{relative}"]
        assert outcome["shards"][relative] == sha256_file(str(path)), relative
        rows = json.loads(path.read_text(encoding="utf-8"))["rows"]
        assert not [row for row in rows if "arm" in row]  # l'étiquette naît à l'ingestion
    assert not (base / "arm_sensitivity_raw_raw.json").exists()
    assert "arm_sensitivity_raw_raw.json" not in outcome["shards"]
    # publications de la sensibilité : dans `results.json` (§4.4), pas dans un shard séparé
    results = json.loads((base / "results.json").read_text(encoding="utf-8"))
    assert results["arms"]["sensitivity_fp32"]["arm"] == "sensitivity_fp32"
    assert results["arms"]["sensitivity_fp32"]["n"] == 6
    assert not (base / "arm_sensitivity.json").exists()
    # l'ancrage garde son shard étiqueté §7.2, avec le hash du brut relevé avant ingestion
    labeled = json.loads((base / "arm_anchor.json").read_text(encoding="utf-8"))
    assert labeled["arm"] == "anchor"
    assert labeled["raw_sha256_before_ingestion"] == outcome["shards"]["arm_anchor_raw.json"]
    assert all(row["arm"] == "anchor" for row in labeled["records"])


def test_r_j18r_08_dry_run_plan_declares_the_shards_of_each_arm() -> None:
    plan = j18r.dry_run_plan(
        ["--output-dir", "_artefact/j18r_decision_tests/plan", "--blocks", "2", "--bootstrap-iterations", "25"]
    )
    arms = plan["arms"]
    assert arms["anchor"]["shards"] == ["arm_anchor_raw.json", "arm_anchor.json"]
    assert arms["sensitivity_fp32"]["shards"] == ["arm_sensitivity_raw.json"]
    assert arms["control"]["shards"] == ["arm_control.json"]


# ---------------------------------------------------------------------------
# R-J18R-09 — propagation de `--output-dir` aux bras délégués, chaîne figée au canonique
# ---------------------------------------------------------------------------

#: §7.3b du PRE, chaîne d'ancrage figée (recopiée littéralement, non reconstruite).
PRE_ANCHOR_INVOCATION = (
    "python modules/dopd/harness/diagnostic_annexe_d.py --model Qwen/Qwen3-0.6B --device cuda "
    "--dtype bfloat16 --prompt-file modules/dopd/eval/annexe_d_qwen3_0.6b_multiprompt/prompts.json "
    "--blocks 100 --seed 20261023 --bootstrap-seed 20261024 --local-files-only "
    "--output modules/dopd/eval/j18r_decision/arm_anchor_raw.json --top-k 16"
)


def test_r_j18r_09_canonical_invocations_stay_frozen_at_the_default_output() -> None:
    """D3 : au `--output-dir` canonique, la chaîne d'ancrage reste celle du PRE §7.3b à
    l'identique ; la sensibilité n'en diffère que par `--dtype float32` et le nom de sortie."""
    assert j18r.anchor_invocation(j18r.DEFAULT_OUTPUT_DIR) == PRE_ANCHOR_INVOCATION
    sensitivity = j18r.sensitivity_invocation(j18r.DEFAULT_OUTPUT_DIR)
    assert sensitivity == PRE_ANCHOR_INVOCATION.replace("--dtype bfloat16", "--dtype float32").replace(
        "arm_anchor_raw.json", "arm_sensitivity_raw.json"
    )


def test_r_j18r_09_delegated_invocations_write_under_the_requested_output_dir() -> None:
    """D3 : un vol d'essai hors du chemin canonique écrit ses bruts sous son `--output-dir`."""
    out = str(_scratch_output_dir("invocations"))
    anchor_tokens = shlex.split(j18r.anchor_invocation(out, blocks=2))
    sensitivity_tokens = shlex.split(j18r.sensitivity_invocation(out, blocks=2))
    for tokens, raw in ((anchor_tokens, "arm_anchor_raw.json"), (sensitivity_tokens, "arm_sensitivity_raw.json")):
        output = tokens[tokens.index("--output") + 1]
        assert output == f"{out}/{raw}"
        assert j18r._REPO_ROOT / output == j18r._REPO_ROOT / out / raw
    assert anchor_tokens[anchor_tokens.index("--blocks") + 1] == "2"
    assert "modules/dopd/eval/j18r_decision" not in " ".join(anchor_tokens)


# ---------------------------------------------------------------------------
# R-J18R-10 — bras délégué défaillant : INVALIDE, pas de crash, pas de STOP sur un C9 inévaluable
# ---------------------------------------------------------------------------


def test_r_j18r_10_missing_raw_shard_degrades_to_invalide_without_crash(monkeypatch) -> None:
    """Un sous-processus J15 qui ne produit pas son shard : `execute` ne lève pas, ne publie
    pas `results.json` (§7.2), l'issue est INVALIDE ; les bras exécutés vont au bout."""
    env = _fake_execute_environment(monkeypatch, returncode=1, write_raw=False)
    out = _scratch_output_dir("anchor_failed")
    argv = ["--output-dir", str(out), "--blocks", "2", "--bootstrap-iterations", "25"]
    outcome = j18r.execute(j18r.build_parser().parse_args(argv), argv)
    assert outcome["results_written"] is False
    assert not (j18r._REPO_ROOT / out / "results.json").exists()
    assert outcome["decision"]["status"] == "INVALIDE"
    assert "bras requis incomplet" in outcome["decision"]["failed_criteria"]
    assert "C9" in outcome["decision"]["failed_criteria"]
    assert env["arms"] == ["control", "decision"]
    assert (j18r._REPO_ROOT / out / "arm_control.json").exists()


def test_r_j18r_10_unevaluable_c9_is_invalide_not_stop() -> None:
    """Aucune ligne partagée avec J15 n'est une dérive mesurée : INVALIDE (corriger et
    relancer), pas STOP — la priorité 1 est réservée à un ancrage mesuré en échec (§4.5)."""
    criteria = {key: True for key in j18r.CRITERIA_KEYS}
    criteria["C9"] = False
    criteria["C9_evaluable"] = False
    criteria["arms_complete"] = True
    criteria["C1"] = True
    decision = j18r.decide_j18r_issue(criteria, {"R_a": False, "R_b": False, "R_c": False})
    assert decision["status"] == "INVALIDE"
    assert decision["priority"] == 2
    assert "C9" in decision["failed_criteria"]


def test_r_j18r_10_measured_c9_failure_still_is_stop() -> None:
    criteria = {key: True for key in j18r.CRITERIA_KEYS}
    criteria["C9"] = False
    criteria["arms_complete"] = True
    decision = j18r.decide_j18r_issue(criteria, {"R_a": False, "R_b": False, "R_c": False})
    assert decision["status"] == "STOP"
    assert decision["priority"] == 1


def test_r_j18r_10_c9_and_publications_survive_an_empty_anchor() -> None:
    reference = {
        "rows": [{"prompt": "p01", "config_index": "N4_V6", "block_index": 0, "position": 1, "noise": 0.0, "delta_kl": 0.08}]
    }
    empty = j18r.criterion_c9([], reference)
    assert empty["holds"] is False
    assert empty["clause_b"]["n_shared_keys"] == 0
    publications = j18r.arm_publications([], arm="anchor", bootstrap_seed=5, iterations=25)
    assert publications == {"arm": "anchor", "n": 0}
    c4 = j18r.criterion_c4({"anchor": publications, "control": publications, "decision": publications}, 2, 20)
    assert c4["per_config"] == {}


# ---------------------------------------------------------------------------
# R-J18R-11 — chemin complet de `execute` exercé : ingestion, shards, sidecar, hashes
# ---------------------------------------------------------------------------


def test_r_j18r_11_execute_writes_every_produced_file_and_the_sidecar_hashes_them(monkeypatch) -> None:
    """§7.2 : `lineage.json` porte le sha256 de chaque sortie, recalculable fichier par fichier."""
    env = _fake_execute_environment(monkeypatch)
    out = _scratch_output_dir("execute_end_to_end")
    argv = ["--output-dir", str(out), "--blocks", "2", "--bootstrap-iterations", "25", "--device", "cpu"]
    outcome = j18r.execute(j18r.build_parser().parse_args(argv), argv)
    base = j18r._REPO_ROOT / out
    produced = sorted(
        [
            "arm_anchor_raw.json",
            "arm_sensitivity_raw.json",
            "arm_anchor.json",
            "arm_control.json",
            "arm_decision.json",
            "results.json",
        ]
    )
    sidecar = json.loads((base / "lineage.json").read_text(encoding="utf-8"))
    assert sorted(sidecar["artifact"]["hashes"]) == produced
    for name in produced:
        assert sidecar["artifact"]["hashes"][name] == sha256_file(str(base / name)), name
    assert outcome["shards"]["lineage.json"] == sha256_file(str(base / "lineage.json"))
    assert outcome["results_written"] is True
    assert env["arms"] == ["control", "decision"]
    # le brut d'ancrage ingéré n'est pas réécrit par J18R
    assert (base / "arm_anchor_raw.json").read_text(encoding="utf-8") == env["raw_payloads"][f"{out}/arm_anchor_raw.json"]


# ---------------------------------------------------------------------------
# R-J18R-12 — préfixe du marginal conditionné sur les identifiants du prompt (eq. 14)
# ---------------------------------------------------------------------------


class _SpyScorer:
    """Spy du contrat `TeacherScorer` : enregistre les préfixes reçus et refuse le préfixe vide,
    comme le scorer de production (`teacher_scorer_qwen3.py:288-289` et `:380-381`)."""

    def __init__(self, vocab: int = 12) -> None:
        self.vocab = vocab
        self.job_prefixes: list[tuple[int, ...]] = []
        self.replay_prefixes: list[tuple[int, ...]] = []

    def score(self, request) -> object:
        prefix = tuple(int(token) for token in request.prefix_ids)
        self.job_prefixes.append(prefix)
        if not prefix:
            raise ValueError("prefix_ids est vide : le prior causal exige au moins un token")
        return types.SimpleNamespace(
            prior=np.full(self.vocab, -6.0, dtype=np.float64),
            candidate_loglikelihoods=np.full((len(request.candidate_ids), len(request.suffix_ids)), -6.0),
            reference_loglikelihood=-6.0,
        )

    def replay_job(self, prefix_ids, candidate_ids, suffix_ids):
        prefix = tuple(int(token) for token in prefix_ids)
        self.replay_prefixes.append(prefix)
        if not prefix:
            raise ValueError("prefix_ids est vide")
        return np.full(self.vocab, -6.0, dtype=np.float64), np.full((len(candidate_ids), len(suffix_ids)), -6.0)


def test_r_j18r_12_provider_conditions_the_block_prefix_on_prompt_ids() -> None:
    """D4 : `marginal_for_position` passe le préfixe **relatif** du bloc ; à `position = 0` il est
    vide et le scorer de production le refuse. Le provider conditionne sur `q` (eq. 14), comme le
    support J15 (`model_forward` forward `prompt_ids + prefix`)."""
    spy = _SpyScorer()
    provider = j18r.ProductionScoreProvider(spy, prompt_ids=(7, 8, 9))
    prior, scores, reference = provider.job(prefix=(), position=0, tokens=[3, 4], suffix=(5,), reference_token=3)
    assert spy.job_prefixes == [(7, 8, 9)]
    assert prior.shape == (spy.vocab,)
    assert scores.shape == (2, 1)
    assert reference == -6.0
    replayed_prior, replayed_scores = provider.replay_job(prefix=(), candidates=[3, 4], suffix=(5,))
    assert spy.replay_prefixes == [(7, 8, 9)]
    # §5.2 : le contrat du provider rend le prior **aux candidats** (le scorer de production
    # rend le prior plein vocabulaire, eq. 14).
    assert replayed_prior.shape == (2,)
    assert replayed_prior.tolist() == [-6.0, -6.0]
    provider.job(prefix=(1,), position=1, tokens=[3], suffix=(5,), reference_token=3)
    assert spy.job_prefixes[-1] == (7, 8, 9, 1)


def test_r_j18r_12_provider_factory_is_per_prompt_and_binds_device_by_keyword() -> None:
    devices: list[str] = []
    spies: list[_SpyScorer] = []

    def scorer_class(model, **kwargs) -> _SpyScorer:
        devices.append(str(kwargs["device"]))
        spy = _SpyScorer()
        spies.append(spy)
        return spy

    factory = j18r.production_provider_factory(scorer_class, _FakeModel(), device="cpu")
    first = factory((7, 8, 9))
    second = factory((10, 11))
    first.job(prefix=(1,), position=1, tokens=[3], suffix=(5,), reference_token=3)
    second.job(prefix=(), position=0, tokens=[3], suffix=(5,), reference_token=3)
    assert devices == ["cpu", "cpu"]
    assert spies[0].job_prefixes == [(7, 8, 9, 1)]
    assert spies[1].job_prefixes == [(10, 11)]


# ---------------------------------------------------------------------------
# R-J18R-13 — `prompt_token_ids` : rendu texte puis encodage (D4)
# ---------------------------------------------------------------------------


class _MappingEncoding(Mapping):
    """`Mapping` qui n'est **pas** un `dict` (comme `BatchEncoding` : `UserDict`)."""

    def __init__(self, **values: object) -> None:
        self._values = dict(values)

    def __getitem__(self, key: str) -> object:
        return self._values[key]

    def __iter__(self):
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)


class _FakeChatTokenizer:
    """Tokenizer factice dont `apply_chat_template` rend un **texte** puis renvoie un
    `Mapping` (non-`dict`) — la forme qui faisait échouer le code précédent."""

    def __init__(self, ids: list[int]) -> None:
        self.ids = list(ids)
        self.rendered: list[str] = []
        self.chat_kwargs: list[dict] = []

    def apply_chat_template(self, messages, **kwargs):
        self.chat_kwargs.append(dict(kwargs))
        rendered = "<|im_start|>user\n" + messages[0]["content"] + "<|im_end|>\n"
        self.rendered.append(rendered)
        return rendered

    def __call__(self, text, add_special_tokens=True, return_tensors=None):
        assert text in self.rendered
        assert add_special_tokens is False
        return _MappingEncoding(input_ids=list(self.ids), attention_mask=[1] * len(self.ids))


class _FakeTorch:
    """Minimal `torch` pour `prompt_token_ids` : `tensor` rend une liste d'entiers."""

    long = "long"

    @staticmethod
    def tensor(values, dtype=None, device=None):
        return list(values)


def test_r_j18r_13_prompt_token_ids_accepts_a_mapping_that_is_not_a_dict(monkeypatch) -> None:
    """D4 : le retour `apply_chat_template`/`tokenizer` est un `Mapping` (pas un `dict`) ;
    le code précédent itérait sur ses **clés** et levait `ValueError: 'input_ids'`."""
    monkeypatch.setattr(j18r, "_torch", lambda: _FakeTorch)
    tokenizer = _FakeChatTokenizer([11, 22, 33])
    ids = j18r.prompt_token_ids(tokenizer, "hello", "cpu", chat_template=True)
    assert ids == [11, 22, 33]
    # le gabarit est rendu **en texte** (`tokenize=False`) puis encodé sans jetons spéciaux
    assert tokenizer.chat_kwargs == [
        {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
    ]


@pytest.mark.parametrize(
    "encoded",
    [
        [1, 2, 3],
        [[1, 2, 3]],
        _MappingEncoding(input_ids=[1, 2, 3]),
    ],
)
def test_r_j18r_13_extract_input_ids_accepts_the_allowed_shapes(encoded) -> None:
    assert j18r.extract_input_ids(encoded) == [1, 2, 3]


@pytest.mark.parametrize(
    "encoded",
    [
        [],
        [[1, 2], [3, 4]],
        [1, "2"],
        [1, True],
        _MappingEncoding(attention_mask=[1, 2]),
        "abc",
        42,
    ],
)
def test_r_j18r_13_extract_input_ids_refuses_ambiguous_or_invalid_shapes(encoded) -> None:
    with pytest.raises(ValueError):
        j18r.extract_input_ids(encoded)


def _local_chat_tokenizer():
    """Tokenizer local (aucun réseau) au gabarit de chat inspiré de Qwen3."""
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    root = j18r._REPO_ROOT / "_artefact" / "j18r_decision_tests" / "tokenizer"
    root.mkdir(parents=True, exist_ok=True)
    vocab = {
        "[UNK]": 0,
        "[PAD]": 1,
        "<|im_start|>": 2,
        "<|im_end|>": 3,
        "<think>": 4,
        "</think>": 5,
        "user": 6,
        "assistant": 7,
        "\n": 8,
        "Question:": 9,
        "Answer:": 10,
    }
    backend = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    template = (
        "{%- if messages[0]['role'] == 'user' %}"
        "{{ '<|im_start|>user\\n' + messages[0]['content'] + '<|im_end|>\\n<|im_start|>assistant\\n' }}"
        "{%- endif %}"
        "{%- if enable_thinking is defined and enable_thinking is false %}"
        "{{ '<think>\\n\\n</think>\\n\\n' }}"
        "{%- endif %}"
    )
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, chat_template=template
    )
    tokenizer.save_pretrained(str(root))
    return transformers.AutoTokenizer.from_pretrained(str(root))


def test_r_j18r_13_real_tokenizer_ids_match_the_direct_tokenize_path(monkeypatch) -> None:
    """Le chemin texte→encodage donne **exactement** les ids de
    `apply_chat_template(tokenize=True, return_dict=False)` (Qwen3, `transformers` 5.x)."""
    monkeypatch.setattr(j18r, "_torch", lambda: _FakeTorch)
    tokenizer = _local_chat_tokenizer()
    messages = [{"role": "user", "content": "Question: Answer:"}]
    direct = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    ids = j18r.prompt_token_ids(tokenizer, "Question: Answer:", "cpu", chat_template=True)
    assert ids == list(direct)


def test_r_j18r_13_real_tokenizer_disables_thinking(monkeypatch) -> None:
    """`enable_thinking=False` est effectif : le rendu porte le bloc de réflexion **vide**."""
    monkeypatch.setattr(j18r, "_torch", lambda: _FakeTorch)
    tokenizer = _local_chat_tokenizer()
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Question: Answer:"}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    assert "<think>" in rendered and "</think>" in rendered
    thinking_block = rendered.split("<think>", 1)[1].split("</think>", 1)[0]
    assert thinking_block.strip() == ""


# ---------------------------------------------------------------------------
# R-J18R-14 — N6 non décisionnel : budget, exception, ancrage INVALIDE vs STOP
# ---------------------------------------------------------------------------


def test_r_j18r_14_n6_budget_exceeded_publishes_partial_and_keeps_the_main_decision(
    monkeypatch,
) -> None:
    """Plafond N6 atteint avec observations valides : publication partielle explicite,
    décision principale conservée (C1–C10 inchangée)."""
    env = _fake_execute_environment(monkeypatch)
    original_run_arm = j18r.run_arm

    def fake_run_arm(**kwargs):
        arm = str(kwargs["arm"])
        env["arms"].append(arm)
        if arm != "n6_chat_template":
            return original_run_arm(**kwargs)
        observations = kwargs.get("observations_out")
        if observations is not None:
            observations.append(_observation_for(arm))
        raise j18r.BudgetExceeded("time", "budget temps dépassé (1801.0 s > 1800.0 s)")

    monkeypatch.setattr(j18r, "run_arm", fake_run_arm)
    out = _scratch_output_dir("n6_budget")
    argv = [
        "--output-dir",
        str(out),
        "--blocks",
        "2",
        "--bootstrap-iterations",
        "25",
        "--n6-chat-template",
    ]
    outcome = j18r.execute(j18r.build_parser().parse_args(argv), argv)
    assert outcome["results_written"] is True
    results = json.loads((j18r._REPO_ROOT / out / "results.json").read_text(encoding="utf-8"))
    assert results["n6"]["status"] == "budget_exceeded"
    assert results["n6"]["partial"] is True
    assert results["n6"]["decisionnal"] is False
    assert results["n6"]["error"]["kind"] == "time"
    assert results["n6"]["n_observations"] == 1
    # la décision principale ne dépend pas de N6
    assert results["decision"]["status"] in ("INVALIDE", "NO-GO", "GO", "GO conditionnel", "STOP")
    assert "n6_chat_template" not in results["criteria"]["C4"]["per_config"]


def _observation_for(arm: str):
    """Observation minimale (constructeur du harnais) pour un bras non décisionnel."""
    return j18r.MarginalObservation(
        arm=arm,
        prompt="p01",
        config_index="N4_V12",
        block_index=0,
        position=0,
        noise=0.0,
        visible_future=True,
        causal_kl=0.5,
        future_kl=0.4,
        delta_kl=0.1,
        full_kl=0.4,
        reason=j18r.REASON_APPLIED,
        n_candidates=4,
    )


def test_r_j18r_14_n6_exception_before_any_observation_invents_no_statistics(monkeypatch) -> None:
    """Échec N6 **avant toute observation** : aucune statistique inventée, état consigné,
    le run reste valide (bras requis complets)."""
    env = _fake_execute_environment(monkeypatch)
    original_run_arm = j18r.run_arm

    def fake_run_arm(**kwargs):
        arm = str(kwargs["arm"])
        env["arms"].append(arm)
        if arm != "n6_chat_template":
            return original_run_arm(**kwargs)
        raise RuntimeError("crash N6")

    monkeypatch.setattr(j18r, "run_arm", fake_run_arm)
    out = _scratch_output_dir("n6_crash")
    argv = [
        "--output-dir",
        str(out),
        "--blocks",
        "2",
        "--bootstrap-iterations",
        "25",
        "--n6-chat-template",
    ]
    outcome = j18r.execute(j18r.build_parser().parse_args(argv), argv)
    assert outcome["results_written"] is True
    results = json.loads((j18r._REPO_ROOT / out / "results.json").read_text(encoding="utf-8"))
    assert results["n6"]["status"] == "failed"
    assert results["n6"]["n_observations"] == 0
    assert results["arms"]["n6_chat_template"]["n"] == 0
    assert results["decision"]["status"] != "INVALIDE" or "n6" not in str(results["decision"])


def test_r_j18r_14_n6_observations_then_exception_are_diagnostic_only(monkeypatch) -> None:
    """Observations puis exception : échec consigné, données **diagnostiques** seulement."""
    env = _fake_execute_environment(monkeypatch)
    original_run_arm = j18r.run_arm

    def fake_run_arm(**kwargs):
        arm = str(kwargs["arm"])
        env["arms"].append(arm)
        if arm != "n6_chat_template":
            return original_run_arm(**kwargs)
        observations = kwargs.get("observations_out")
        if observations is not None:
            observations.append(_observation_for(arm))
        raise RuntimeError("crash N6 après observation")

    monkeypatch.setattr(j18r, "run_arm", fake_run_arm)
    out = _scratch_output_dir("n6_crash_after")
    argv = [
        "--output-dir",
        str(out),
        "--blocks",
        "2",
        "--bootstrap-iterations",
        "25",
        "--n6-chat-template",
    ]
    j18r.execute(j18r.build_parser().parse_args(argv), argv)
    results = json.loads((j18r._REPO_ROOT / out / "results.json").read_text(encoding="utf-8"))
    assert results["n6"]["status"] == "failed"
    assert results["n6"]["error"]["type"] == "RuntimeError"
    assert results["n6"]["n_observations"] == 1
    assert results["arms"]["n6_chat_template"]["partial"] is True
    assert results["arms"]["n6_chat_template"]["diagnostic_only"] is True


def test_r_j18r_14_anchor_execution_failure_is_invalide_not_stop(monkeypatch) -> None:
    """Défaillance d'**exécution** de l'ancrage → INVALIDE (pas STOP) : l'écart C9 n'est pas
    mesuré, le run n'est pas attribuable à une dérive du harnais."""
    env = _fake_execute_environment(monkeypatch, returncode=1, write_raw=False)
    out = _scratch_output_dir("anchor_failed_r14")
    argv = ["--output-dir", str(out), "--blocks", "2", "--bootstrap-iterations", "25"]
    outcome = j18r.execute(j18r.build_parser().parse_args(argv), argv)
    assert outcome["decision"]["status"] == "INVALIDE"
    assert outcome["decision"]["priority"] == 2
    assert "C9" in outcome["decision"]["failed_criteria"]
    assert env["arms"] == ["control", "decision"]


def _c9_reference_and_records(delta: float) -> tuple[dict, list[dict]]:
    """Référence J15 minimale et lignes d'ancrage **alignées** sur la clé
    `(prompt, config_index, block_index, position, noise)`."""
    reference = {
        "rows": [
            {
                "prompt": "p01",
                "config_index": "N4_V6",
                "block_index": 0,
                "position": 0,
                "noise": 0.0,
                "delta_kl": j18r.C9_MEAN_REFERENCE,
            }
        ]
    }
    records = [_row("anchor", "p01", "N4_V6", 0, 0.0, delta)]
    records[0]["position"] = 0
    return reference, records


def test_r_j18r_14_anchor_measured_drift_is_stop() -> None:
    """Écart C9 **mesuré** sur un échantillon conforme au PRE (100 blocs) → STOP."""
    reference, anchor_records = _c9_reference_and_records(j18r.C9_MEAN_REFERENCE + 0.5)
    c9 = j18r.criterion_c9(
        anchor_records, reference, blocks=j18r.CANONICAL_BLOCKS, anchor_executed=True
    )
    assert c9["evaluable"] is True
    assert c9["evaluable_reason"] == "measured"
    assert c9["holds"] is False
    criteria = {key: True for key in j18r.CRITERIA_KEYS}
    criteria["C9"] = False
    criteria["C9_evaluable"] = True
    criteria["arms_complete"] = True
    decision = j18r.decide_j18r_issue(criteria, {"R_a": False, "R_b": False, "R_c": False})
    assert decision["status"] == "STOP"
    assert decision["priority"] == 1


def test_r_j18r_14_smoke_blocks_are_not_a_measured_drift() -> None:
    """Un vol d'essai (2 blocs) ne peut pas conclure à une dérive : C9 n'est pas évaluable
    → l'issue reste INVALIDE (effectifs insuffisants), jamais STOP."""
    reference, anchor_records = _c9_reference_and_records(j18r.C9_MEAN_REFERENCE + 0.5)
    c9 = j18r.criterion_c9(anchor_records, reference, blocks=2, anchor_executed=True)
    assert c9["evaluable"] is False
    assert c9["evaluable_reason"] == "insufficient_blocks"
    criteria = {key: True for key in j18r.CRITERIA_KEYS}
    criteria["C9"] = False
    criteria["C9_evaluable"] = False
    criteria["arms_complete"] = True
    decision = j18r.decide_j18r_issue(criteria, {"R_a": False, "R_b": False, "R_c": False})
    assert decision["status"] == "INVALIDE"
    assert decision["priority"] == 2


def test_r_j18r_14_n6_does_not_enter_c4() -> None:
    """N6 est hors C4 : un effectif N6 insuffisant n'invalide pas le run."""
    records = [_row("decision", "p01", "N4_V12", index, 0.0, 0.1) for index in range(3)]
    publications = {
        "decision": j18r.arm_publications(
            j18r.observations_from_records(records), arm="decision", bootstrap_seed=5, iterations=25
        ),
        "n6_chat_template": j18r.arm_publications(
            j18r.observations_from_records(records[:1]),
            arm="n6_chat_template",
            bootstrap_seed=5,
            iterations=25,
        ),
    }
    c4 = j18r.criterion_c4(publications, blocks=2, n_prompts=20)
    assert "n6_chat_template/N4_V12" not in c4["per_config"]
    assert c4["holds"] is False  # le bras décision a moins de 2000 marginaux : C4 échoue bien


def test_r_j18r_14_global_budget_is_paused_during_n6() -> None:
    """Le budget global « hors N6 » ne court pas pendant N6 (PRE §7.8)."""
    clock = [0.0]
    guard = j18r.BudgetGuard(max_seconds=10.0, clock=lambda: clock[0], rss=lambda: 0.0)
    clock[0] = 5.0
    guard.pause()
    clock[0] = 5000.0
    assert guard.elapsed_seconds == 5.0
    guard.resume()
    assert guard.elapsed_seconds == 5.0
    clock[0] = 5006.0
    assert guard.elapsed_seconds == 11.0


def test_r_j18r_14_n6_time_budget_is_typed_and_global_ones_are_not_intercepted() -> None:
    """Le plafond propre de N6 est typé `"time"` ; les plafonds globaux (`rss`, `gpu`) et le
    temps global (`"global_time"`) restent des arrêts du run."""
    n6 = j18r.BudgetGuard(max_seconds=1.0, clock=lambda: 0.0, rss=lambda: 0.0, time_kind="time")
    with pytest.raises(j18r.BudgetExceeded) as excinfo:
        n6.check(now=2.0)
    assert excinfo.value.kind == "time"

    global_guard = j18r.BudgetGuard(
        max_seconds=1.0, clock=lambda: 0.0, rss=lambda: 0.0, time_kind="global_time"
    )
    with pytest.raises(j18r.BudgetExceeded) as excinfo:
        global_guard.check(now=2.0)
    assert excinfo.value.kind == "global_time"

    rss_guard = j18r.BudgetGuard(max_seconds=100.0, clock=lambda: 0.0, rss=lambda: 100 * 1024**3)
    with pytest.raises(j18r.BudgetExceeded) as excinfo:
        rss_guard.check(now=0.0)
    assert excinfo.value.kind == "rss"
