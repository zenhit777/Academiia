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

import errno
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


class ChatTemplateRejected(Exception):
    """Échec **métier** de N6 (gabarit de conversation refusé) : interceptable par le garde."""


class OutOfMemoryError(Exception):
    """Nom de type identique à `torch.cuda.OutOfMemoryError` : échec d'**infrastructure**."""


def _fake_execute_environment(
    monkeypatch, *, returncode: int = 0, write_raw: bool = True, raise_for: str | None = None
) -> dict[str, object]:
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
    # J18R-final : le watchdog VRAM et le preflight sont **simulés** (pas de
    # `nvidia-smi` en CI) ; la lecture est **factice mais mesurée** par le fake.
    monkeypatch.setattr(
        j18r.VramWatchdog,
        "query",
        staticmethod(lambda: {"used_mib": 1_292, "free_mib": 30_896, "total_mib": 32_607}),
    )

    def raw_rows(arm: str) -> list[dict]:
        rows = [_row(arm, f"p{p + 1:02d}", "N4_V6", b, 0.0, 0.05) for p in range(2) for b in range(3)]
        for row in rows:
            row.pop("arm")  # J15 n'étiquette pas : J18R étiquette à l'ingestion
        return rows

    def fake_subprocess(invocation, root, **kwargs):
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
        return {
            "returncode": int(returncode),
            "group": {
                "pgid": 0,
                "limit_mib": int(kwargs.get("limit_mib", 0)),
                "limit_applied": True,
                "stop_after_seconds": float(kwargs.get("stop_after_seconds", 0.0)),
                "stop_signal": None,
                "survivors_after_stop": 0,
                "verified_after_stop": True,
                "elapsed_seconds": 0.0,
            },
            "resisted": False,
        }

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
        # R-J18R-17 : le nombre de **jobs** du bras est publié (base de la couverture de
        # rejeu §5.2) ; le fake l'égale à son nombre d'observations.
        result = {"arm": arm, "records": records, "publications": publications_for(arm, records), "complete": True}
        result["n_jobs"] = len(records)
        if arm == raise_for:
            # échec **métier** du bras non décisionnel : le garde N6 doit l'absorber,
            # **sans** le faire entrer dans C5 ni dans l'issue (R-J18R-15).
            raise ChatTemplateRejected("gabarit de conversation refusé par le professeur")
        return result

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


# ---------------------------------------------------------------------------
# R-J18R-17 — compteur de jobs **du bras** (§3.5), niveaux de bruit du contrôle (§2.5)
# ---------------------------------------------------------------------------


def _config_records(
    arm: str,
    block_size: int,
    vocab_width: int,
    blocks: int,
    prompt: str,
    counter: j18r.JobCounter,
    replay_stats: j18r.ReplayStats,
    integrity: j18r.IntegrityStats,
) -> list:
    """Une configuration sur la fixture NumPy, avec le compteur **partagé** du bras."""
    table = fixture_table()
    support = j18r.build_economical_support(
        lambda prefixes: np.stack([table(prefix) for prefix in prefixes]), block_size, vocab_width, option="A"
    )
    sequences, log_joint = support.enumerate_joint()
    return j18r.process_configuration(
        arm=arm,
        prompt=prompt,
        config_index=f"N{block_size}_V{vocab_width}",
        support=support,
        sequences=sequences,
        log_joint=log_joint,
        provider=FixtureScorer(table),
        seed=j18r.SEED_DECISION,
        blocks=blocks,
        noise_levels=j18r.CONTROL_NOISE_LEVELS if arm == "control" else j18r.NOISE_LEVELS,
        integrity=integrity,
        replay_stats=replay_stats,
        job_counter=counter,
        block_sample=list(range(blocks)),
        future_is_state=(arm == "control"),
    )


def test_r_j18r_17_control_arm_runs_only_the_zero_noise_level() -> None:
    """PRE §2.5 : σ = 0 est donné **au seul** bras de contrôle ; §1.5 fige 18 marginaux."""
    counter = j18r.JobCounter()
    integrity, replay = j18r.IntegrityStats(), j18r.ReplayStats()
    records = []
    for prompt in j18r.CONTROL_PROMPTS:
        for block_size, vocab_width in ((3, 3), (3, 4)):
            records.extend(
                _config_records("control", block_size, vocab_width, len(j18r.CONTROL_BLOCKS), prompt, counter, replay, integrity)
            )
    assert j18r.CONTROL_NOISE_LEVELS == (0.0,)
    assert len(j18r.NOISE_LEVELS) == 4
    assert [observation.noise for observation in records] == [0.0] * len(records)
    assert len(records) == len(j18r.CONTROL_PROMPTS) * len(j18r.CONTROL_BLOCKS) * len(j18r.CONTROL_CONFIGS) == 18
    control = j18r.control_metrics([observation.record() for observation in records])
    assert control["n"] == 18 and control["holds"] is True
    assert control["max_future_kl"] <= control["threshold_max"]


def test_r_j18r_17_job_counter_is_monotone_across_the_configurations_of_a_arm() -> None:
    """Le compteur est **du bras** : il ne repart pas à 1 à chaque configuration (§3.5)."""
    counter = j18r.JobCounter()
    indexes: list[int] = []
    # contrôle : un seul niveau de bruit, une seule position masquée par bloc →
    # 1 job par bloc, donc 10 jobs par configuration, comptés en continu.
    for block_size, vocab_width in ((2, 2), (2, 3)):
        observations = _config_records(
            "control", block_size, vocab_width, 10, "p01", counter, j18r.ReplayStats(), j18r.IntegrityStats()
        )
        indexes.extend([observation.job_index for observation in observations])
    assert len(indexes) == 20
    assert indexes == list(range(1, 21))
    assert counter.value == 20


def test_r_j18r_17_replay_reaches_five_percent_of_the_arm_jobs() -> None:
    """20 jobs d'un bras → 1 job rejoué (index 20) = 5 % ; 18 jobs → 0 et C5(c) échoue.

    La géométrie de contrôle (σ = 0, **une** position masquée par bloc) donne 1 job
    par bloc : le nombre de jobs est donc **exactement** celui des blocs.
    """
    counter, replay = j18r.JobCounter(), j18r.ReplayStats()
    twenty = _config_records("control", 2, 2, 20, "p01", counter, replay, j18r.IntegrityStats())
    assert len(twenty) == 20
    assert replay.n_jobs == 1
    assert replay.violations == []
    c5 = j18r.criterion_c5(
        j18r.IntegrityStats(), replay, {"holds": True}, arm_jobs={"control": 20, "decision": 20}
    )
    assert c5["clause_c_replay"]["sampled_jobs"] == 20
    assert c5["clause_c_replay"]["expected_replays"] == 1
    assert c5["clause_c_replay"]["coverage"] == 0.05
    assert c5["clause_c_replay"]["holds"] is True
    # un bras à 18 jobs ne **peut** atteindre 5 % : le critère échoue, il n'est pas faked.
    replay18 = j18r.ReplayStats()
    _config_records("control", 2, 2, 18, "p01", j18r.JobCounter(), replay18, j18r.IntegrityStats())
    assert replay18.n_jobs == 0
    c5_small = j18r.criterion_c5(
        j18r.IntegrityStats(), replay18, {"holds": True}, arm_jobs={"control": 18, "decision": 18}
    )
    assert c5_small["clause_c_replay"]["holds"] is False


def test_r_j18r_17_expected_replays_is_ceiling_of_five_percent() -> None:
    """L'attendu est `ceil(5 % × jobs)` : 1 rejeu sur 24 jobs ne suffit pas."""
    stats = j18r.ReplayStats()
    stats.record_job(20, 0.0)
    c5 = j18r.criterion_c5(j18r.IntegrityStats(), stats, {"holds": True}, arm_jobs={"decision": 24})
    assert c5["clause_c_replay"]["expected_replays"] == 2
    assert c5["clause_c_replay"]["holds"] is False
    stats.record_job(40, 0.0)
    c5 = j18r.criterion_c5(j18r.IntegrityStats(), stats, {"holds": True}, arm_jobs={"decision": 24})
    assert c5["clause_c_replay"]["coverage"] == 2 / 24
    assert c5["clause_c_replay"]["holds"] is True


# ---------------------------------------------------------------------------
# R-J18R-15 / R-J18R-16 — portée de C5, garde N6 à trois étages
# ---------------------------------------------------------------------------


def test_r_j18r_15_c5_is_evaluated_on_its_own_arms_and_n6_is_isolated() -> None:
    """C5 observe **contrôle + décision** ; l'intégrité N6 vit dans ses propres instances."""
    run_replay = j18r.ReplayStats()
    run_replay.record_job(20, 0.0)
    n6_replay = j18r.ReplayStats()
    n6_replay.record_job(20, 5.0)  # anomalie N6 : écart très au-dessus de la tolérance
    c5_run = j18r.criterion_c5(
        j18r.IntegrityStats(), run_replay, {"holds": True}, arm_jobs={"control": 18, "decision": 20}
    )
    c5_n6 = j18r.criterion_c5(
        j18r.IntegrityStats(),
        n6_replay,
        {"holds": True},
        observed_arms=("n6_chat_template",),
        excluded_arms=("control", "decision", "anchor", "sensitivity_fp32"),
        arm_jobs={"n6_chat_template": 20},
    )
    assert c5_run["observed_arms"] == ["control", "decision"]
    assert "n6_chat_template" in c5_run["excluded_arms"]
    assert c5_run["clause_c_replay"]["violations"] == []
    assert c5_run["clause_c_replay"]["holds"] is True
    assert c5_run["clause_e_control_and_enumeration"]["holds"] is True
    # clause D : **aucune** cible float32 observée sur cette instance vide → agrégat faux,
    # la violation N6 (dans l'autre instance) n'y est **pas** entrée.
    assert c5_run["clause_d_float32_applied"]["holds"] is False
    assert c5_run["holds"] is False
    assert c5_n6["observed_arms"] == ["n6_chat_template"]
    assert c5_n6["clause_c_replay"]["violations"] == [20]
    assert c5_n6["holds"] is False


def test_r_j18r_16_infra_and_business_failures_are_distinguished() -> None:
    """La liste d'infrastructure est **étroite** : un échec métier n'y entre pas."""
    assert j18r.is_infra_failure(OutOfMemoryError("CUDA: out of memory")) is True
    assert j18r.is_infra_failure(OutOfMemoryError("")) is True
    assert j18r.is_infra_failure(RuntimeError("cannot allocate 40 GiB")) is True
    enom = OSError()
    enom.errno = errno.ENOMEM
    assert j18r.is_infra_failure(enom) is True
    assert j18r.is_infra_failure(ChatTemplateRejected("gabarit refusé")) is False
    assert j18r.is_infra_failure(ValueError("k < V obligatoire")) is False


def test_r_j18r_16_the_n6_guard_absorbs_business_failure_and_propagates_the_rest(monkeypatch) -> None:
    """Trois étages : temps N6 → `partial` ; RSS / GPU / mémoire → **propagés**."""
    args = types.SimpleNamespace(bootstrap_seed=j18r.BOOTSTRAP_SEED_DECISION)
    kwargs = {
        "args": args,
        "prompts": [],
        "model": None,
        "provider_factory": None,
        "tokenizer": None,
        "device": "cpu",
        "budget": j18r.BudgetGuard(max_seconds=1800.0, clock=lambda: 0.0, rss=lambda: 0.0, time_kind="time"),
        "integrity": j18r.IntegrityStats(),
        "replay_stats": j18r.ReplayStats(),
    }

    def _raise(arm, **_kwargs):
        raise ChatTemplateRejected("gabarit de conversation refusé par le professeur")
    monkeypatch.setattr(j18r, "run_arm", _raise)
    absorbed = j18r.run_arm_guarded("n6_chat_template", **kwargs)
    assert absorbed["status"] == "failed" and absorbed["complete"] is False
    assert absorbed["publications"]["partial"] is True
    assert absorbed["publications"]["diagnostic_only"] is False
    assert absorbed["n_jobs"] == 0
    assert absorbed["error"]["type"] == "ChatTemplateRejected"

    def _raise_infra(arm, **_kwargs):
        raise OutOfMemoryError("CUDA: out of memory")

    monkeypatch.setattr(j18r, "run_arm", _raise_infra)
    with pytest.raises(OutOfMemoryError):
        j18r.run_arm_guarded("n6_chat_template", **kwargs)

    def _raise_rss(arm, **_kwargs):
        raise j18r.BudgetExceeded("rss", "plafond RSS global atteint pendant N6")

    monkeypatch.setattr(j18r, "run_arm", _raise_rss)
    with pytest.raises(j18r.BudgetExceeded) as excinfo:
        j18r.run_arm_guarded("n6_chat_template", **kwargs)
    assert excinfo.value.kind == "rss"

    def _raise_time(arm, **_kwargs):
        raise j18r.BudgetExceeded("time", "plafond propre de N6 (1800 s)")

    monkeypatch.setattr(j18r, "run_arm", _raise_time)
    partial = j18r.run_arm_guarded("n6_chat_template", **kwargs)
    assert partial["status"] == "budget_exceeded" and partial["publications"]["partial"] is True


def test_r_j18r_15_an_n6_anomaly_does_not_move_the_run_issue(monkeypatch) -> None:
    """Une anomalie N6 reste **dans le bloc N6** : `C5` et l'issue sont inchangés."""
    out_without = _scratch_output_dir("n6_anomaly_without")
    argv = ["--output-dir", str(out_without), "--blocks", "2", "--bootstrap-iterations", "25"]
    env_without = _fake_execute_environment(monkeypatch)
    outcome_without = j18r.execute(j18r.build_parser().parse_args(argv), argv)
    monkeypatch.undo()
    env = _fake_execute_environment(monkeypatch, raise_for="n6_chat_template")
    out_with = _scratch_output_dir("n6_anomaly_with")
    argv_with = ["--output-dir", str(out_with), "--blocks", "2", "--bootstrap-iterations", "25", "--n6-chat-template"]
    outcome_with = j18r.execute(j18r.build_parser().parse_args(argv_with), argv_with)
    base = j18r._REPO_ROOT
    without = json.loads((base / out_without / "results.json").read_text(encoding="utf-8"))
    with_n6 = json.loads((base / out_with / "results.json").read_text(encoding="utf-8"))
    assert env_without["arms"] == ["control", "decision"]
    assert outcome_without["results_written"] is True
    assert env["arms"] == ["control", "decision", "n6_chat_template"]
    # C5 du run : contrôle + décision **uniquement**, aucune violation N6.
    assert with_n6["criteria"]["C5"]["observed_arms"] == ["control", "decision"]
    assert "n6_chat_template" in with_n6["criteria"]["C5"]["excluded_arms"]
    assert with_n6["criteria"]["C5"]["clause_c_replay"]["violations"] == []
    assert with_n6["criteria"]["C5"] == without["criteria"]["C5"]
    # l'issue est **identique** : elle n'est pas fonction du statut N6.
    assert with_n6["decision"] == without["decision"]
    assert outcome_with["results_written"] is True
    # l'anomalie est publiée **dans le bloc N6**, avec son intégrité propre.
    n6 = with_n6["n6"]
    assert n6["present"] is True and n6["decisionnal"] is False
    assert n6["status"] == "failed" and n6["partial"] is True
    assert n6["error"]["type"] == "ChatTemplateRejected"
    assert n6["integrity_c5"]["observed_arms"] == ["n6_chat_template"]
    assert n6["integrity_c5"]["clause_c_replay"]["holds"] is False
    assert "jamais dans le C5 du run" in n6["integrity_note"]
    assert without["n6"]["present"] is False
    # R-J18R-17 : le dénominateur est le nombre de **jobs** publié par les bras.
    assert with_n6["criteria"]["C5"]["arm_jobs"] == {"control": 6, "decision": 6}


# ---------------------------------------------------------------------------
# R-J18R-18 — portée réelle du plafond RSS, mesure des sous-processus (constat D9)
# ---------------------------------------------------------------------------


def test_r_j18r_18_child_memory_is_measured_after_a_real_subprocess() -> None:
    """`RUSAGE_CHILDREN` mesure le pic d'un enfant **attendu** par le harnais."""
    subprocess.run([sys.executable, "-c", "x = bytearray(200 * 1024 * 1024); print(len(x))"], check=True)
    observation = j18r.delegated_memory_observation()
    assert observation["observed"] is True
    assert observation["max_rss_children_bytes"] > 0
    assert "RUSAGE_CHILDREN" in observation["measured_by"]
    assert "ne sont **pas** couverts" in observation["note"]


def test_r_j18r_18_budget_scope_is_declared_in_the_plan_and_in_the_run(monkeypatch) -> None:
    """La portée du plafond est **consignée** : le plafond n'est pas devenu « sans plafond »."""
    plan = j18r.dry_run_plan(["--output-dir", "_artefact/j18r_decision_tests/plan_scope", "--blocks", "2"])
    scope = plan["budgets"]["rss_budget_scope"]
    assert "processus du harnais" in scope and "n'est pas couverte" in scope
    assert plan["arms"]["control"]["control_noise_levels"] == [0.0]
    assert plan["arms"]["decision"]["control_noise_levels"] is None
    env = _fake_execute_environment(monkeypatch)
    out = _scratch_output_dir("delegated_memory")
    argv = ["--output-dir", str(out), "--blocks", "2", "--bootstrap-iterations", "25"]
    outcome = j18r.execute(j18r.build_parser().parse_args(argv), argv)
    results = json.loads((j18r._REPO_ROOT / out / "results.json").read_text(encoding="utf-8"))
    # R-J18R-17 : le dénominateur de la couverture est le nombre de **jobs** publié par
    # chaque bras (`n_jobs`), pas un nombre d'observations deviné.
    assert results["criteria"]["C5"]["arm_jobs"] == {"control": 6, "decision": 6}
    assert env["arms"] == ["control", "decision"]
    runtime = results["runtime"]
    assert "processus du harnais" in runtime["rss_budget_scope"]
    assert "sous-processus" in runtime["rss_budget_scope"]
    delegated = runtime["delegated_memory"]["anchor"]
    assert {"max_rss_children_bytes", "cgroup_memory_peak_bytes", "observed", "measured_by", "note"} <= set(delegated)
    assert results["protocol"]["control_noise_levels"] == [0.0]
    assert results["numerical_integrity"]["observed_arms"] == ["control", "decision"]
    assert "n6_chat_template" in results["numerical_integrity"]["excluded_arms"]
    assert outcome["manifest"]["verification"]["budget_scope"] == runtime["rss_budget_scope"]


# ---------------------------------------------------------------------------
# R-J18R-19 — audit : engagement aux frontières de bloc, couverture N6, inventaire
# complet C9, C6 complet, écriture gardée par le temps, sous-processus supervisés
# (points A1, A4, A5, A8, D9 de la revue)
# ---------------------------------------------------------------------------


def _reference_rows() -> list[dict]:
    """Lignes J15 gelées (référence C9), sans étiquette `arm`."""
    path = j18r._REPO_ROOT / "modules" / "dopd" / "eval" / "annexe_d_qwen3_0.6b_multiprompt" / "results.json"
    return json.loads(path.read_text(encoding="utf-8"))["rows"]


def _group_fake(rows_by_arm: Mapping[str, list[dict]], *, resisted: bool = False):
    """Sous-processus simulé conforme au contrat supervisé D9 (groupe, RLIMIT, stop)."""
    def fake(invocation: str, root: Path, **kwargs) -> dict:
        tokens = shlex.split(invocation)
        output = root / Path(tokens[tokens.index("--output") + 1])
        arm = {
            j18r.RAW_SHARDS["anchor"]: "anchor",
            j18r.RAW_SHARDS["sensitivity_fp32"]: "sensitivity_fp32",
        }[output.name]
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"rows": rows_by_arm[arm]}, indent=2) + "\n", encoding="utf-8")
        return {
            "returncode": 0,
            "group": {
                "pgid": 4242,
                "limit_mib": int(kwargs["limit_mib"]),
                "limit_applied": True,
                "stop_after_seconds": float(kwargs["stop_after_seconds"]),
                "stop_signal": "SIGKILL" if resisted else None,
                "survivors_after_stop": 2 if resisted else 0,
                "verified_after_stop": not resisted,
                "elapsed_seconds": 1.0,
                "cgroup_memory_peak_bytes": 12345.0,
            },
            "resisted": resisted,
        }
    return fake


def test_r_j18r_19_block_commit_shares_only_complete_blocks() -> None:
    """A1/D5 : un bloc **complet** est engagé sur la liste partagée **à sa frontière** ;
    le bloc où le budget tombe n'est **jamais** engagé partiellement."""
    table = fixture_table()
    block_size, vocab_width, blocks = 3, 3, 3
    support = j18r.build_economical_support(
        lambda prefixes: np.stack([table(prefix) for prefix in prefixes]), block_size, vocab_width, option="A"
    )
    sequences, log_joint = support.enumerate_joint()
    shared: list = []
    calls = {"n": 0}

    class _Budget:
        def check(self) -> dict:
            calls["n"] += 1
            if calls["n"] == 2:
                raise j18r.BudgetExceeded("time", "plafond atteint à la frontière de bloc")
            return {"elapsed_seconds": 0.0, "rss_gib": 0.0, "gpu_peak_mib": 0.0}

    try:
        j18r.process_configuration(
            arm="n6_chat_template",
            prompt="p01",
            config_index=f"N{block_size}_V{vocab_width}",
            support=support,
            sequences=sequences,
            log_joint=log_joint,
            provider=FixtureScorer(table),
            seed=j18r.SEED_DECISION,
            blocks=blocks,
            noise_levels=[0.0],
            integrity=j18r.IntegrityStats(),
            replay_stats=j18r.ReplayStats(),
            job_counter=j18r.JobCounter(),
            block_sample=list(range(blocks)),
            future_is_state=False,
            observations_out=shared,
            budget=_Budget(),
        )
    except j18r.BudgetExceeded:
        pass
    assert calls["n"] == 2
    assert shared
    assert {int(o.block_index) for o in shared} == {0, 1}


def test_r_j18r_19_n6_coverage_is_published_when_partial(monkeypatch) -> None:
    """A1 : budget N6 → publication **partielle**, stats **conservées**, couverture
    **réellement engagée** publiée (blocs complets sur la liste partagée)."""
    env = _fake_execute_environment(monkeypatch)
    original_run_arm = j18r.run_arm

    def fake_run_arm(**kwargs):
        arm = str(kwargs["arm"])
        env["arms"].append(arm)
        if arm != "n6_chat_template":
            return original_run_arm(**kwargs)
        observations = kwargs.get("observations_out")
        if observations is not None:
            observations.extend(_observation_for(arm) for _ in range(2))
        raise j18r.BudgetExceeded("time", "budget temps dépassé (1801.0 s > 1800.0 s)")

    monkeypatch.setattr(j18r, "run_arm", fake_run_arm)
    out = _scratch_output_dir("n6_coverage_partial")
    argv = [
        "--output-dir", str(out), "--blocks", "2", "--bootstrap-iterations", "25",
        "--n6-chat-template",
    ]
    outcome = j18r.execute(j18r.build_parser().parse_args(argv), argv)
    results = json.loads((j18r._REPO_ROOT / out / "results.json").read_text(encoding="utf-8"))
    assert results["n6"]["status"] == "budget_exceeded"
    assert results["n6"]["partial"] is True
    assert results["n6"]["n_observations"] == 2
    assert results["n6"]["coverage"]["n_blocks_completed"] == 1
    assert results["n6"]["coverage"]["n_jobs_emitted"] == 0
    # les statistiques tirées des observations engagées restent publiées
    assert results["arms"]["n6_chat_template"]["n"] == 2
    assert results["arms"]["n6_chat_template"]["diagnostic_only"] is True
    assert outcome["results_written"] is True


def test_r_j18r_19_n6_coverage_is_zero_before_any_observation(monkeypatch) -> None:
    """A1 : exception **avant toute observation** → couverture nulle, aucune statistique
    inventée, le run reste valide (bras requis complets)."""
    env = _fake_execute_environment(monkeypatch)
    original_run_arm = j18r.run_arm

    def fake_run_arm(**kwargs):
        arm = str(kwargs["arm"])
        env["arms"].append(arm)
        if arm != "n6_chat_template":
            return original_run_arm(**kwargs)
        raise RuntimeError("crash N6 avant la première observation")

    monkeypatch.setattr(j18r, "run_arm", fake_run_arm)
    out = _scratch_output_dir("n6_coverage_zero")
    argv = [
        "--output-dir", str(out), "--blocks", "2", "--bootstrap-iterations", "25",
        "--n6-chat-template",
    ]
    j18r.execute(j18r.build_parser().parse_args(argv), argv)
    results = json.loads((j18r._REPO_ROOT / out / "results.json").read_text(encoding="utf-8"))
    assert results["n6"]["status"] == "failed"
    assert results["n6"]["n_observations"] == 0
    assert results["n6"]["coverage"]["n_blocks_completed"] == 0
    assert results["n6"]["coverage"]["n_jobs_emitted"] == 0
    assert results["arms"]["n6_chat_template"]["n"] == 0


def test_r_j18r_19_fast_path_rechecks_global_time_ceiling() -> None:
    """A5 : le throttle de 1 Hz porte sur l'échantillonnage RSS/GPU, **pas** sur le
    plafond de temps global : `check()` sur chemin rapide voit le plafond."""
    ticks = iter([0.0, 0.0, 0.5, 0.8])
    guard = j18r.BudgetGuard(max_seconds=0.7, clock=lambda: next(ticks))
    first = guard.check()
    assert first["elapsed_seconds"] == 0.0
    second = guard.check()  # chemin rapide (gap < 1 s), écart cumulé 0.5 ≤ 0.7 s
    assert second["elapsed_seconds"] == 0.5
    with pytest.raises(j18r.BudgetExceeded) as exc:
        guard.check()
    assert exc.value.kind == "time"


def test_r_j18r_19_guard_time_uses_injected_clock_and_freezes_during_pause() -> None:
    """A5 : `guard_time` évalue l'horloge **injectée** ; le temps global est **gelé**
    pendant la pause N6, jamais remis à zéro."""
    t = [0.0]
    guard = j18r.BudgetGuard(max_seconds=10.0, clock=lambda: t[0])
    t[0] = 4.0
    assert guard.guard_time("écriture") == 4.0
    guard.pause()
    t[0] = 9000.0
    assert guard.guard_time("écriture") == 4.0
    guard.resume()
    t[0] = 9005.0
    assert guard.guard_time("écriture") == 9.0
    t[0] = 9011.0
    with pytest.raises(j18r.BudgetExceeded) as exc:
        guard.guard_time("écriture")
    assert exc.value.kind == "time"


def test_r_j18r_19_results_not_written_when_global_time_exceeds_before_write(
    monkeypatch,
) -> None:
    """A5 : dépassement du plafond global **avant** la persistance → `results.json`
    **non écrit** ; les shards d'observations restent acquis."""
    _fake_execute_environment(monkeypatch)
    out = _scratch_output_dir("global_time_write")
    argv = ["--output-dir", str(out), "--blocks", "2", "--bootstrap-iterations", "25"]
    class _Guard(j18r.BudgetGuard):
        def check(self) -> dict:
            return {"elapsed_seconds": 0.0, "rss_gib": 0.0, "gpu_peak_mib": 0.0}

        def guard_time(self, label: str) -> float:
            raise j18r.BudgetExceeded("global_time", f"dépassement du plafond global avant écriture ({label})")
    monkeypatch.setattr(j18r, "BudgetGuard", _Guard)
    with pytest.raises(j18r.BudgetExceeded) as exc:
        j18r.execute(j18r.build_parser().parse_args(argv), argv)
    assert exc.value.kind == "global_time"
    assert not (j18r._REPO_ROOT / out / "results.json").exists()
    assert (j18r._REPO_ROOT / out / j18r.ARM_SHARDS["control"]).exists()
    assert (j18r._REPO_ROOT / out / j18r.ARM_SHARDS["decision"]).exists()


def test_r_j18r_19_full_anchor_inventory_is_measured_drift() -> None:
    """A4 : inventaire complet (20 prompts × 3 configs × 100 indices distincts) →
    écart **mesuré**, `evaluable` vrai, dérive → STOP (clause a en échec)."""
    rows = _reference_rows()
    drifted = [
        dict(r, causal_kl=float(r["causal_kl"]) + 0.05) if index % 2 == 0 else dict(r)
        for index, r in enumerate(rows)
    ]
    c9 = j18r.criterion_c9(drifted, {"rows": rows}, blocks=100, anchor_executed=True, n_prompts=20)
    assert c9["evaluable"] is True
    assert c9["evaluable_reason"] == "measured"
    assert c9["sample_complete"] is True
    assert c9["missing_pairs"] == [] and c9["incomplete_pairs"] == []
    assert c9["clause_a"]["holds"] is False
    assert c9["clause_b"]["holds"] is True
    assert c9["holds"] is False


def test_r_j18r_19_truncated_inventory_is_invalidate_not_drift() -> None:
    """A4 : un shard d'ancrage **tronqué** rend C9 **non évaluable** ; la clause (b)
    n'interprète **pas** la troncature comme une dérive."""
    rows = _reference_rows()
    complete = j18r.criterion_c9(rows, {"rows": rows}, blocks=100, anchor_executed=True, n_prompts=20)
    truncated = [r for r in rows if r["prompt"] != "p03"]
    c9 = j18r.criterion_c9(truncated, {"rows": rows}, blocks=100, anchor_executed=True, n_prompts=20)
    assert c9["evaluable"] is False
    assert c9["evaluable_reason"] == "incomplete_anchor_sample"
    assert c9["missing_pairs"] == [["p03", "N4_V12"], ["p03", "N4_V6"], ["p03", "N6_V6"]]
    # l'écart sur les clés partagées n'est **pas** gonflé par la troncature
    assert c9["clause_b"]["mean_abs_gap_shared"] == complete["clause_b"]["mean_abs_gap_shared"]
    assert c9["clause_b"]["holds"] is True
    assert c9["holds"] is True
    # `STOP` est réservé à un écart **mesuré** en échec ; l'inventaire incomplet tombe en INVALIDE
    issue = j18r.decide_j18r_issue(
        {"C1": False, "C4": False, "C5": True, "C6": True, "C7": True, "C8": True,
         "C9": True, "C10": True, "arms_complete": True, "C9_evaluable": False,
         "C1_sigma_positive": False},
        {},
    )
    assert issue["status"] == "INVALIDE"
    assert issue["priority"] == 2


def test_r_j18r_19_full_inventory_drift_gives_stop_at_run_level(monkeypatch) -> None:
    """A4 : au niveau du run, dérive mesurée sur échantillon **conforme** → STOP
    (priorité 1), `results.json` écrit, bras complets."""
    _fake_execute_environment(monkeypatch)
    rows = _reference_rows()
    drifted = [
        dict(r, causal_kl=float(r["causal_kl"]) + 0.05) if index % 2 == 0 else dict(r)
        for index, r in enumerate(rows)
    ]
    monkeypatch.setattr(
        j18r, "run_arm_subprocess", _group_fake({"anchor": drifted, "sensitivity_fp32": rows})
    )
    out = _scratch_output_dir("stop_full_inventory")
    argv = ["--output-dir", str(out), "--blocks", "100", "--bootstrap-iterations", "25"]
    outcome = j18r.execute(j18r.build_parser().parse_args(argv), argv)
    results = json.loads((j18r._REPO_ROOT / out / "results.json").read_text(encoding="utf-8"))
    c9 = results["criteria"]["C9"]
    assert c9["evaluable"] is True
    assert c9["evaluable_reason"] == "measured"
    assert c9["sample_complete"] is True
    assert c9["holds"] is False
    assert results["criteria"]["arms_complete"] is True
    assert outcome["decision"]["status"] == "STOP"
    assert outcome["decision"]["priority"] == 1
    assert outcome["results_written"] is True


def test_r_j18r_19_truncated_shard_gives_invalidate_at_run_level(monkeypatch) -> None:
    """A4 : au niveau du run, shard d'ancrage tronqué → INVALIDE (non dérive),
    `C9_evaluable` faux, effet **non** mesuré."""
    _fake_execute_environment(monkeypatch)
    rows = _reference_rows()
    truncated = [r for r in rows if r["prompt"] != "p03"]
    monkeypatch.setattr(
        j18r, "run_arm_subprocess", _group_fake({"anchor": truncated, "sensitivity_fp32": truncated})
    )
    out = _scratch_output_dir("invalidate_truncated")
    argv = ["--output-dir", str(out), "--blocks", "100", "--bootstrap-iterations", "25"]
    outcome = j18r.execute(j18r.build_parser().parse_args(argv), argv)
    results = json.loads((j18r._REPO_ROOT / out / "results.json").read_text(encoding="utf-8"))
    c9 = results["criteria"]["C9"]
    assert c9["evaluable"] is False
    assert c9["evaluable_reason"] == "incomplete_anchor_sample"
    assert c9["sample_complete"] is False
    assert outcome["decision"]["status"] == "INVALIDE"
    assert "C4" in outcome["decision"]["failed_criteria"]


def test_r_j18r_19_c6_compares_all_fields_and_keeps_acceptance_flag_in_trace() -> None:
    """A8 : C6 compare la **configuration complète** (modèle, device, dtype par bras,
    fichier, effectifs, graines, rejeu, plafonds, sortie) ; la trace fidèle garde
    `--confirm-accepted`, la commande **rejouable** ne le porte jamais."""
    canonical_argv = shlex.split(j18r.CANONICAL_COMMAND)[2:]
    argv = canonical_argv + ["--confirm-accepted"]
    args = j18r.build_parser().parse_args(argv)
    c6 = j18r.criterion_c6(argv, args)
    assert c6["holds"] is True
    assert all(bool(field["holds"]) for field in c6["fields"].values())
    assert len(c6["fields"]) == len(j18r.CANONICAL_FIELDS)
    assert "--confirm-accepted" in c6["command_executed"]
    assert c6["string_match_without_acceptance_flag"] is True
    reconstruction = c6["canonical_reconstruction"]
    assert "--confirm-accepted" not in reconstruction
    parsed = j18r.build_parser().parse_args(shlex.split(reconstruction)[2:])
    for name, _flag in j18r.CANONICAL_FIELDS:
        assert getattr(parsed, name) == getattr(args, name)


def test_r_j18r_19_c6_fails_on_one_configured_field() -> None:
    """A8 : `holds` n'est **pas** acquis par convention — un seul champ en écart
    (`--device cpu` vs `cuda` figé §7.3) rend C6 faux."""
    argv = shlex.split(j18r.CANONICAL_COMMAND)[2:] + ["--device", "cpu", "--confirm-accepted"]
    args = j18r.build_parser().parse_args(argv)
    c6 = j18r.criterion_c6(argv, args)
    assert c6["holds"] is False
    assert c6["fields"]["device"]["holds"] is False
    assert c6["fields"]["model"]["holds"] is True
    assert c6["string_match_without_acceptance_flag"] is False


def test_r_j18r_19_manifest_carries_training_and_the_faithful_trace(monkeypatch) -> None:
    """A8 : le manifeste consigne le bloc `training` **mesuré** au runtime, la garde
    de temps **avant** écriture, et la commande exécutée **avec** le drapeau."""
    _fake_execute_environment(monkeypatch)
    out = _scratch_output_dir("manifest_training")
    argv = ["--output-dir", str(out), "--blocks", "2", "--bootstrap-iterations", "25", "--confirm-accepted"]
    outcome = j18r.execute(j18r.build_parser().parse_args(argv), argv)
    verification = outcome["manifest"]["verification"]
    training = verification["training"]
    assert training["performed"] is False
    assert training["backward"] is False
    assert training["weights_updated"] is False
    assert training["backward_count"] == 0
    assert training["requires_grad_all_false"] is True
    assert verification["time_guarded_before_write"] is True
    assert "--confirm-accepted" in outcome["manifest"]["producer"]["command"]


def test_r_j18r_19_plan_declares_delegated_limits_and_scope() -> None:
    """D9 : le plan dry-run consigne le plafond RLIMIT_AS délégué, le plafond de stop
    et la **portée réelle** (groupe, descendants, GPU couvert par le watchdog)."""
    plan = j18r.dry_run_plan(["--output-dir", "_artefact/j18r_decision_tests/plan_d9", "--blocks", "2"])
    budgets = plan["budgets"]
    assert budgets["budget_rss_mib"] == int(budgets["max_rss_gib"] * 1024)
    # J18R-final : le plafond RLIMIT_AS délégué est **virtuel** et mesuré ; il est
    # distinct du plafond RSS du processus du harnais. Le plan ne doit plus
    # publier une valeur qui diverge de celle appliquée par le code.
    assert budgets["delegated_memory_limit_mib"] == j18r.memory_limit_mib(
        j18r.build_parser().parse_args(["--output-dir", "x"])
    )
    assert budgets["delegated_memory_limit_mib"] == int(j18r.DELEGATED_RLIMIT_AS_GIB * 1024)
    assert budgets["delegated_memory_limit_mib"] != budgets["budget_rss_mib"]
    assert budgets["delegated_stop_after_seconds"] == budgets["max_seconds"]
    scope = budgets["rss_budget_scope"]
    assert "RLIMIT_AS" in scope
    assert "processus du harnais" in scope
    assert "watchdog VRAM" in scope


def test_r_j18r_19_memory_limit_mechanism_is_available() -> None:
    """D9 : le mécanisme du plafond est **mesuré**, pas deviné (child qui pose et
    relit RLIMIT_AS avant `exec`)."""
    assert j18r.memory_limit_available() >= 64
    args = j18r.build_parser().parse_args(["--output-dir", "x"])
    assert j18r.memory_limit_mib(args) == int(j18r.DELEGATED_RLIMIT_AS_GIB * 1024)


def test_r_j18r_19_rlimit_is_applied_before_exec_in_the_group() -> None:
    """D9 : RLIMIT_AS est posé **avant** `exec` sur le child ; le child relit sa
    limite **avant** toute allocation."""
    out = _scratch_output_dir("rlimit_probe")
    probe = j18r._REPO_ROOT / out / "probe.py"
    result = j18r._REPO_ROOT / out / "result.json"
    probe.write_text(
        "import json, pathlib, resource\n"
        "soft, hard = resource.getrlimit(resource.RLIMIT_AS)\n"
        "pathlib.Path('" + str(out) + "/result.json').write_text("
        "json.dumps({'soft': soft, 'hard': hard}))\n",
        encoding="utf-8",
    )
    res = j18r.run_arm_subprocess(
        f"{sys.executable} {out}/probe.py", j18r._REPO_ROOT, limit_mib=64, stop_after_seconds=30.0
    )
    assert res["returncode"] == 0
    group = res["group"]
    assert group["limit_applied"] is True
    assert group["limit_mib"] == 64
    payload = json.loads(result.read_text(encoding="utf-8"))
    assert payload["soft"] == 64 * 1024 * 1024
    assert payload["hard"] == 64 * 1024 * 1024
    assert group["verified_after_stop"] is True


def test_r_j18r_19_enforced_limit_refusal_stops_before_the_run(monkeypatch) -> None:
    """D9 : un plafond **non appliqué** est refusé **avant** le run complet
    (sortie 2), `results.json` **non écrit**."""
    _fake_execute_environment(monkeypatch)
    out = _scratch_output_dir("rlimit_refusal")

    def refusing(invocation, root, **kwargs):
        raise j18r.MemoryLimitEnforced("plafond RLIMIT_AS non appliqué")

    monkeypatch.setattr(j18r, "run_arm_subprocess", refusing)
    argv = ["--output-dir", str(out), "--blocks", "2", "--bootstrap-iterations", "25"]
    with pytest.raises(SystemExit) as exc:
        j18r.execute(j18r.build_parser().parse_args(argv), argv)
    assert exc.value.code == 2
    assert not (j18r._REPO_ROOT / out / "results.json").exists()


def test_r_j18r_19_resisted_group_is_never_recorded_complete(monkeypatch) -> None:
    """D9 : des membres qui **résistent** au stop rendent le bras **non complet** ;
    le run est INVALIDE, `results.json` non écrit."""
    _fake_execute_environment(monkeypatch)
    out = _scratch_output_dir("resisted_group")
    rows = _reference_rows()
    monkeypatch.setattr(
        j18r, "run_arm_subprocess",
        _group_fake({"anchor": rows, "sensitivity_fp32": rows}, resisted=True),
    )
    argv = ["--output-dir", str(out), "--blocks", "2", "--bootstrap-iterations", "25"]
    outcome = j18r.execute(j18r.build_parser().parse_args(argv), argv)
    assert outcome["results_written"] is False
    assert not (j18r._REPO_ROOT / out / "results.json").exists()
    assert outcome["decision"]["status"] == "INVALIDE"
    assert "bras requis incomplet" in outcome["decision"]["failed_criteria"]
    assert (j18r._REPO_ROOT / out / j18r.RAW_SHARDS["anchor"]).exists()


def test_r_j18r_19_stop_signal_is_sent_to_the_group_on_time_ceiling() -> None:
    """D9 : au-delà du plafond de **temps**, `SIGTERM` au **groupe** ; vérification
    après stop, `resisted` faux si le groupe est intégralement arrêté."""
    res = j18r.run_arm_subprocess(
        f"{sys.executable} -c \"import time; time.sleep(30)\"",
        j18r._REPO_ROOT,
        limit_mib=64,
        stop_after_seconds=0.2,
    )
    assert res["returncode"] < 0
    group = res["group"]
    assert group["stop_signal"] in ("SIGTERM", "SIGKILL")
    assert group["survivors_after_stop"] == 0
    assert group["verified_after_stop"] is True
    assert res["resisted"] is False


def test_r_j18r_19_group_observation_publishes_limit_stop_and_verification() -> None:
    """D9 : la mémoire déléguée publiée porte la limite posée, le signal de stop,
    la vérification après stop, la couverture du groupe (members covered)."""
    group = {
        "pgid": 4242,
        "limit_mib": 64,
        "limit_applied": True,
        "stop_after_seconds": 14400.0,
        "stop_signal": "SIGTERM",
        "survivors_after_stop": 0,
        "verified_after_stop": True,
        "elapsed_seconds": 1.0,
        "cgroup_memory_peak_bytes": 12345.0,
    }
    obs = j18r.delegated_memory_observation(group)
    assert obs["group_members_covered"] is True
    assert obs["limit_mib"] == 64
    assert obs["stop_after_seconds"] == 14400.0
    assert obs["group_stop_signal"] == "SIGTERM"
    assert obs["group_verified_after_stop"] is True
    assert obs["cgroup_memory_peak_bytes"] == 12345.0
    assert "RLIMIT_AS" in obs["measured_by"]


# ---------------------------------------------------------------------------
# J18R-final — reprise post-OOM : watchdog VRAM, exclusion mutuelle, RLIMIT
# ---------------------------------------------------------------------------


def test_j18r_final_vram_watchdog_records_the_hard_threshold_event(monkeypatch) -> None:
    """Watchdog : au-delà du **seuil dur**, l'événement est consigné (seuil, observé,
    free, total, index d'échantillon) ; le watchdog ne lève rien par lui-même."""
    readings = [
        {"used_mib": 1_000, "free_mib": 31_000, "total_mib": 32_000},
        {"used_mib": 28_673, "free_mib": 3_327, "total_mib": 32_000},
    ]
    monkeypatch.setattr(j18r.VramWatchdog, "query", staticmethod(lambda: readings.pop(0)))
    watchdog = j18r.VramWatchdog(max_used_mib=28_672, sample_seconds=0.0)
    watchdog.sample()
    assert watchdog.exceeded is False
    watchdog.sample()
    assert watchdog.exceeded is True
    event = watchdog.event
    assert event is not None
    assert event["reason"] == "vram_used_above_hard_threshold"
    assert event["threshold_used_mib"] == 28_672
    assert event["observed_used_mib"] == 28_673
    assert event["observed_free_mib"] == 3_327
    assert event["observed_total_mib"] == 32_000
    assert event["sample_index"] == 2
    trace = watchdog.to_dict()
    assert trace["exceeded"] is True
    assert trace["peak_used_mib"] == 28_673
    assert trace["min_free_mib"] == 3_327
    assert trace["threshold_used_mib"] == 28_672


def test_j18r_final_vram_watchdog_unavailable_is_refused(monkeypatch) -> None:
    """Instrument indisponible ⇒ **refus** (pas de supposition de budget)."""
    def boom() -> dict[str, int]:
        raise j18r.VramWatchdogUnavailable("nvidia-smi absent")

    monkeypatch.setattr(j18r.VramWatchdog, "query", staticmethod(boom))
    watchdog = j18r.VramWatchdog()
    with pytest.raises(j18r.VramWatchdogUnavailable):
        watchdog.sample()


def test_j18r_final_vram_preflight_refuses_when_free_vram_is_below_threshold(monkeypatch) -> None:
    """PRE §7.8 : VRAM libre < 17 GiB ⇒ STOP, aucun run lancé."""
    monkeypatch.setattr(
        j18r.VramWatchdog,
        "query",
        staticmethod(lambda: {"used_mib": 20_000, "free_mib": 10_000, "total_mib": 32_607}),
    )
    with pytest.raises(j18r.VramPreflightRefused):
        j18r.vram_preflight()


def test_j18r_final_vram_preflight_accepts_sufficient_free_vram(monkeypatch) -> None:
    """PRE §7.8 : VRAM libre suffisante ⇒ le preflight rend la mesure mesurée."""
    monkeypatch.setattr(
        j18r.VramWatchdog,
        "query",
        staticmethod(lambda: {"used_mib": 1_292, "free_mib": 30_896, "total_mib": 32_607}),
    )
    reading = j18r.vram_preflight()
    assert reading["free_mib"] == 30_896
    assert reading["total_mib"] == 32_607


def test_j18r_final_vram_watchdog_kills_the_process_group(monkeypatch) -> None:
    """Watchdog branché : le dépassement du seuil **tue le groupe** et l'arrêt est
    typé (`BudgetExceeded.kind == "gpu_vram"`) avec le fait mesuré."""
    monkeypatch.setattr(
        j18r.VramWatchdog,
        "query",
        staticmethod(lambda: {"used_mib": 30_000, "free_mib": 2_000, "total_mib": 32_000}),
    )
    watchdog = j18r.VramWatchdog(max_used_mib=28_672, sample_seconds=0.0)
    with pytest.raises(j18r.BudgetExceeded) as excinfo:
        j18r.run_arm_subprocess(
            f"{sys.executable} -c \"import time; time.sleep(30)\"",
            j18r._REPO_ROOT,
            limit_mib=64,
            stop_after_seconds=30.0,
            watchdog=watchdog,
            group_rss_max_bytes=0,
        )
    assert excinfo.value.kind == "gpu_vram"
    assert excinfo.value.event is not None
    assert excinfo.value.event["reason"] == "vram_used_above_hard_threshold"
    assert excinfo.value.event["observed_used_mib"] == 30_000


def test_j18r_final_group_rss_limit_kills_the_process_group(monkeypatch) -> None:
    """Plafond RSS de groupe : dépassement ⇒ kill du groupe, `kind == "memory"`."""
    with pytest.raises(j18r.BudgetExceeded) as excinfo:
        j18r.run_arm_subprocess(
            f"{sys.executable} -c \"import time; time.sleep(30)\"",
            j18r._REPO_ROOT,
            limit_mib=64,
            stop_after_seconds=30.0,
            watchdog=None,
            group_rss_max_bytes=1,
        )
    assert excinfo.value.kind == "memory"
    assert excinfo.value.event is not None
    assert excinfo.value.event["reason"] == "group_rss_above_hard_threshold"
    assert excinfo.value.event["threshold_bytes"] == 1


def test_j18r_final_measurement_lock_refuses_a_second_concurrent_job() -> None:
    """Sérialisation : le second lancement est **refusé** (pas mis en file)."""
    with j18r.measurement_lock(j18r._REPO_ROOT):
        with pytest.raises(j18r.MeasurementLockBusy):
            with j18r.measurement_lock(j18r._REPO_ROOT):
                pass
    # le verrou est relâché : un lancement suivant passe.
    with j18r.measurement_lock(j18r._REPO_ROOT):
        pass


def test_j18r_final_measurement_lock_refuses_a_second_job_from_another_process() -> None:
    """Sérialisation inter-processus : un second processus est refusé."""
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import fcntl, os, sys, time;"
                " path = sys.argv[1];"
                " fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644);"
                " fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB);"
                " print('locked', flush=True);"
                " time.sleep(30)"
            ),
            str(j18r._REPO_ROOT / j18r.MEASUREMENT_LOCK_RELATIVE_PATH),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout is not None
        assert holder.stdout.readline().strip() == "locked"
        with pytest.raises(j18r.MeasurementLockBusy):
            with j18r.measurement_lock(j18r._REPO_ROOT):
                pass
    finally:
        holder.kill()
        holder.wait(timeout=30)


def test_j18r_final_rlimit_is_recalibrated_on_the_measured_virtual_peak() -> None:
    """RLIMIT_AS recalé : 128 GiB, supérieur au `VmPeak` mesuré (109,73 GiB), et
    la dérive 160 → 128 GiB est documentée (pas de plafond « à vue de nez »)."""
    assert j18r.DELEGATED_RLIMIT_AS_GIB == 128
    derivation = j18r.DELEGATED_RLIMIT_AS_DERIVATION
    assert "109,73 GiB" in derivation
    assert "16,6 %" in derivation
    assert "48–64 GiB est écartée" in derivation


def test_j18r_final_execute_publishes_the_watchdog_and_lock_facts(monkeypatch) -> None:
    """Le watchdog branché, le preflight, le verrou et les plafonds délégués sont
    publiés dans `results.runtime` **et** dans le manifeste."""
    _fake_execute_environment(monkeypatch)
    out = _scratch_output_dir("watchdog_published")
    argv = ["--output-dir", str(out), "--blocks", "2", "--bootstrap-iterations", "25"]
    outcome = j18r.execute(j18r.build_parser().parse_args(argv), argv)
    results = json.loads((j18r._REPO_ROOT / out / "results.json").read_text(encoding="utf-8"))
    runtime = results["runtime"]
    assert runtime["vram_watchdog"]["threshold_used_mib"] == j18r.VRAM_WATCHDOG_MAX_USED_MIB
    assert runtime["vram_watchdog"]["exceeded"] is False
    assert runtime["vram_preflight"]["free_mib"] >= j18r.VRAM_PREFLIGHT_MIN_FREE_MIB
    assert runtime["measurement_lock"]["exclusive"] is True
    assert runtime["delegated_rlimit_as"]["limit_gib"] == 128
    assert runtime["delegated_group_rss_limit"]["limit_gib"] == 24
    verification = outcome["manifest"]["verification"]
    assert verification["vram_watchdog"] == runtime["vram_watchdog"]
    assert verification["measurement_lock"] == runtime["measurement_lock"]
    assert verification["delegated_rlimit_as"] == runtime["delegated_rlimit_as"]


def test_j18r_final_frozen_invocation_launches_sys_executable_when_python_is_absent(monkeypatch) -> None:
    """La chaîne figée §7.3b commence par `python` (PRE §7.3b). Si le binaire
    `python` est absent du poste, le child est lancé avec `sys.executable` et la
    **substitution** est consignée ; la chaîne figée comparée au PRE n'est **pas**
    modifiée (C6)."""
    out = _scratch_output_dir("interpreter_substitution")
    root = j18r._REPO_ROOT
    probe = root / out / "probe.py"
    marker = root / out / "marker.txt"
    probe.parent.mkdir(parents=True, exist_ok=True)
    probe.write_text(
        "import pathlib, sys\n"
        "pathlib.Path(sys.argv[1]).write_text('ok', encoding='utf-8')\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(j18r.shutil, "which", lambda command: None)
    invocation = f"python {probe} {marker}"
    res = j18r.run_arm_subprocess(invocation, root, limit_mib=64, stop_after_seconds=30.0)
    assert res["returncode"] == 0
    assert marker.read_text(encoding="utf-8") == "ok"
    assert res["interpreter"]["substituted"] is True
    assert res["interpreter"]["invocation_token"] == "python"
    assert res["interpreter"]["executed_token"] == sys.executable
    assert j18r.anchor_invocation(j18r.DEFAULT_OUTPUT_DIR).startswith("python ")
