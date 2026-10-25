"""Tests CPU de la sémantique du harnais du diagnostic Annexe D (J14).

Ces tests utilisent un modèle **faux**, déterministe et sans paramètre : ils
vérifient l'indexation des logits de suffixe, la construction des cibles
future-aware, le repli causal et l'échantillonnage des états masqués. Aucun
checkpoint n'est chargé et aucun GPU n'est requis.

La preuve de la porte scientifique reste le résultat GPU versionné ; ces tests
couvrent les sémantiques qui, sinon, resteraient non vérifiées.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

# Garde de collecte : sans torch, le module entier est ignore (et non une
# erreur d'import). `torch` est lie par importorskip, pas par `import torch`.
torch = pytest.importorskip("torch", reason="torch absent : tests GPU ignores")

from academiia.dopd.diagnostic import RestrictedSupport  # noqa: E402

_HARNESS_PATH = Path(__file__).resolve().parents[1] / "harness" / "diagnostic_annexe_d.py"
_SPEC = importlib.util.spec_from_file_location("dopd_harness_annexe_d", _HARNESS_PATH)
assert _SPEC is not None and _SPEC.loader is not None
harness = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = harness
_SPEC.loader.exec_module(harness)


class _FakeOutput:
    """Sortie minimale compatible avec ``model(...).logits``."""

    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits


class _FakeModel:
    """Modèle déterministe : le logit en position ``t`` lit le token ``t+1``.

    ``logits[t, v] = 1.0`` si ``v == input_ids[t+1]``, ``0.0`` sinon, plus un
    biais ``bias`` appliqué au token lu. Cette règle permet de vérifier
    l'indexation exacte des positions de logits et des cibles décalées.
    """

    def __init__(self, vocab_size: int = 32, bias: float = 0.5) -> None:
        self.vocab_size = vocab_size
        self.bias = bias
        self.config = type("Config", (), {"vocab_size": vocab_size})()

    def __call__(self, input_ids: torch.Tensor) -> _FakeOutput:
        batch, length = input_ids.shape
        logits = torch.zeros((batch, length, self.vocab_size), dtype=torch.float32)
        next_tokens = torch.zeros((batch, length), dtype=torch.long)
        next_tokens[:, :-1] = input_ids[:, 1:]
        next_tokens[:, -1] = 0
        logits += self.bias * torch.nn.functional.one_hot(
            next_tokens, num_classes=self.vocab_size
        ).to(torch.float32)
        return _FakeOutput(logits)


def _toy_support() -> RestrictedSupport:
    """Support 2 positions x 2 tokens avec des probabilités connues."""
    return RestrictedSupport(
        supports=(np.asarray([10, 11]), np.asarray([20, 21])),
        prefixes=(((),), ((10,), (11,),)),
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


def _prompt_ids(length: int = 3) -> torch.Tensor:
    return torch.tensor([1, 2, 3][:length], dtype=torch.long)


def test_suffix_loglikelihoods_reads_the_token_at_the_next_position():
    """Les scores de suffixe doivent lire le logit de la position du candidat.

    Un décalage d'une position (logit en ``t`` pour le token ``t`` au lieu de
    ``t+1``) change le résultat : le test le détecte.
    """
    model = _FakeModel()
    prompt = _prompt_ids()
    prefix = (10,)
    suffix = (20, 21)
    candidates = np.asarray([10, 11], dtype=np.int64)

    scores = harness._suffix_loglikelihoods(
        model, prompt, prefix, candidates, suffix, torch.device("cpu")
    )

    # Calcul de référence : log_softmax du logit en position candidate+j,
    # appliqué au token suffix[j].
    expected = []
    for candidate in candidates:
        row = torch.cat((prompt, torch.tensor(prefix + (int(candidate),) + suffix)))
        candidate_index = prompt.shape[0] + len(prefix)
        positions = list(range(candidate_index, candidate_index + len(suffix)))
        logits = model(input_ids=row.unsqueeze(0)).logits[0, positions, :]
        log_probs = torch.log_softmax(logits, dim=-1)
        total = 0.0
        for offset, token in enumerate(suffix):
            total += float(log_probs[offset, token])
        expected.append(total)
    np.testing.assert_allclose(scores, np.asarray(expected), atol=1e-6, rtol=0.0)


def test_suffix_loglikelihoods_would_fail_with_a_non_shifted_target():
    """Contrôle de sensibilité : un ciblage non décalé donne un autre score."""
    model = _FakeModel()
    prompt = _prompt_ids()
    prefix = (10,)
    suffix = (20, 21)
    candidates = np.asarray([10, 11], dtype=np.int64)
    scores = harness._suffix_loglikelihoods(
        model, prompt, prefix, candidates, suffix, torch.device("cpu")
    )
    wrong = []
    for candidate in candidates:
        row = torch.cat((prompt, torch.tensor(prefix + (int(candidate),) + suffix)))
        candidate_index = prompt.shape[0] + len(prefix)
        logits = model(input_ids=row.unsqueeze(0)).logits[0, candidate_index, :]
        log_probs = torch.log_softmax(logits, dim=-1)
        wrong.append(float(log_probs[suffix[0]]))
    assert not np.allclose(scores, np.asarray(wrong), atol=1e-6)


def test_future_aware_target_reweights_the_whole_support_and_normalizes():
    """La correction s'applique à tout le support, puis est renormalisée."""
    support = _toy_support()
    model = _FakeModel()
    prompt = _prompt_ids()
    future_block = (10, 20)
    masked = (0,)

    targets = harness._causal_and_future_targets(
        support, masked, future_block, model, prompt, torch.device("cpu")
    )
    causal, future = targets[0]
    np.testing.assert_allclose(causal, [0.6, 0.4], atol=1e-12)
    assert float(future.sum()) == pytest.approx(1.0, abs=1e-12)
    assert np.all(future > 0.0)
    # La référence x̂_i reçoit le poids du prior causal : elle reste dominante
    # quand le futur ne favorise pas le candidat.
    assert future[0] > future[1]


def test_future_aware_target_falls_back_to_causal_without_visible_future():
    """Sans futur visible, la cible future-aware coïncide avec la causale."""
    support = _toy_support()
    model = _FakeModel()
    prompt = _prompt_ids()
    future_block = (10, 20)
    masked = (0, 1)

    targets = harness._causal_and_future_targets(
        support, masked, future_block, model, prompt, torch.device("cpu")
    )
    causal, future = targets[0]
    np.testing.assert_allclose(future, causal, atol=0.0, rtol=0.0)


def test_future_aware_target_falls_back_to_causal_without_suffix():
    """Sans token à droite, la cible future-aware coïncide avec la causale."""
    support = _toy_support()
    model = _FakeModel()
    prompt = _prompt_ids()
    future_block = (10, 20)
    masked = (1,)

    targets = harness._causal_and_future_targets(
        support, masked, future_block, model, prompt, torch.device("cpu")
    )
    causal, future = targets[1]
    np.testing.assert_allclose(future, causal, atol=0.0, rtol=0.0)


def test_masked_positions_masks_at_least_one_position():
    """Une position est toujours masquée, même si tout est retenu au hasard."""
    rng = np.random.default_rng(11)
    for _ in range(50):
        masked = harness._masked_positions(4, rng)
        assert 1 <= len(masked) <= 4
        assert len(set(masked)) == len(masked)
        assert all(0 <= position < 4 for position in masked)


def test_masked_positions_uses_the_retention_probability():
    """Environ 55 % des positions sont masquées avec une rétention de 0,45."""
    rng = np.random.default_rng(3)
    total = 0
    trials = 400
    for _ in range(trials):
        total += len(harness._masked_positions(4, rng))
    ratio = total / (trials * 4)
    assert 0.45 < ratio < 0.65


def test_masked_positions_falls_back_when_everything_is_retained():
    """Si toutes les positions sont retenues, une position est masquée au hasard."""

    class _AllRetained:
        def random(self, size: int) -> np.ndarray:
            # Toutes les valeurs sont sous la probabilité de rétention (0,45).
            return np.zeros(size, dtype=np.float64)

        def integers(self, high: int) -> int:
            return high - 1

    masked = harness._masked_positions(4, _AllRetained())
    assert masked == (3,)


def test_sample_block_returns_a_sequence_of_the_support():
    """L'échantillonnage de bloc renvoie une séquence du support énuméré."""
    support = _toy_support()
    sequences, log_joint = support.enumerate_joint()
    rng = np.random.default_rng(5)
    block = harness._sample_block(support, sequences, log_joint, rng)
    assert len(block) == support.block_size
    assert tuple(block) in {tuple(int(x) for x in row) for row in sequences}
