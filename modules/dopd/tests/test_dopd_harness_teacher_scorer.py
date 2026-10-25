"""Tests du `TeacherScorer` réel d-OPD (J17) — harnais, torch optionnel.

Ces tests n'exercent que la **logique** du scorer sur un modèle factice : ils
ne chargent aucun checkpoint, n'utilisent aucun GPU et ne téléchargent rien.
Le contrat J13 reste vérifié par `test_dopd_interfaces.py` (paquet pur).

Les tests sont ignorés si `torch` n'est pas installé (`pytest.importorskip`) :
la collecte doit passer sans torch (critère C13).

    pytest modules/dopd/tests/test_dopd_harness_teacher_scorer.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

_HARNESS = Path(__file__).resolve().parents[1] / "harness"
if str(_HARNESS) not in sys.path:
    sys.path.insert(0, str(_HARNESS))

from teacher_scorer_qwen3 import Qwen3TeacherScorer, teacher_topk  # noqa: E402

from academiia.dopd import ScoringRequest, TeacherScorer  # noqa: E402

VOCAB_SIZE = 23
HEADS = 2
HEAD_DIM = 4
LAYERS = 2


class _FakeConfig:
    """Configuration minimale compatible avec le calcul du cache."""

    def __init__(self) -> None:
        self.vocab_size = VOCAB_SIZE
        self.hidden_size = HEADS * HEAD_DIM
        self.num_attention_heads = HEADS
        self.num_key_value_heads = HEADS
        self.head_dim = HEAD_DIM
        self.num_hidden_layers = LAYERS


class _FakeCacheLayer:
    """Couche de cache minimal : clés et valeurs par couche."""

    def __init__(self) -> None:
        self.keys = None
        self.values = None

    def get_seq_length(self) -> int:
        return 0 if self.keys is None else int(self.keys.shape[-2])


class _FakeCache:
    """Cache statique : il n'est jamais muté par le forward factice."""

    def __init__(self, layers: list[_FakeCacheLayer]) -> None:
        self.layers = layers

    def batch_repeat_interleave(self, repeats: int) -> None:
        for layer in self.layers:
            if layer.keys is not None:
                layer.keys = layer.keys.repeat_interleave(repeats, dim=0)
                layer.values = layer.values.repeat_interleave(repeats, dim=0)


class _FakeOutput:
    def __init__(self, logits: torch.Tensor, cache: _FakeCache) -> None:
        self.logits = logits
        self.past_key_values = cache


class _FakeModel:
    """Modèle déterministe : la position du candidat change le score.

    Le modèle reproduit les propriétés essentials du contrat :

    * le prior causal est une distribution sur le vocabulaire ;
    * le score de suffixe dépend **du candidat** (sinon le contrôle
      d'exactitude ne testerait rien) ;
    * le chemin cache et le chemin de rejeu donnent le **même** résultat
      logique — c'est ce que le contrôle d'exactitude vérifie sur le vrai
      modèle ;
    * le cache prérempli contient une clé et une valeur par couche.
    """

    def __init__(self) -> None:
        self.config = _FakeConfig()
        self.training = True
        self.calls: list[int] = []
        self.seen_prefixes: list[list[float]] = []
        self._parameter = torch.nn.Parameter(torch.zeros(1), requires_grad=True)

    def parameters(self):
        return iter((self._parameter,))

    def eval(self) -> None:
        self.training = False

    def to(self, device):  # pragma: no cover - interface de compatibilité
        return self

    def _logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Logits déterministes, **indépendants de la position**.

        L'indépendance de la position est volontaire : elle rend le chemin
        cache et le chemin de rejeu **logiquement équivalents**, puisque les
        deux ne voient les mêmes tokens qu'à des positions de contexte
        différentes. L'écart BF16 réel est mesuré sur le vrai modèle.
        """
        token = input_ids.to(torch.float32)
        base = torch.sin(token * 0.37)
        logits = torch.stack(
            [torch.cos(base + offset) for offset in range(VOCAB_SIZE)], dim=-1
        )
        return logits

    def __call__(
        self,
        input_ids: torch.Tensor,
        past_key_values: object | None = None,
        use_cache: bool = True,
        logits_to_keep: int = 0,
    ) -> _FakeOutput:
        self.calls.append(int(input_ids.shape[-1]))
        logits = self._logits(input_ids)
        if logits_to_keep:
            logits = logits[:, -logits_to_keep:, :]
        if past_key_values is None:
            layers = []
            tokens = input_ids[0].to(torch.float32)
            for _ in range(LAYERS):
                layer = _FakeCacheLayer()
                length = int(input_ids.shape[-1])
                layer.keys = torch.zeros(1, HEADS, length, HEAD_DIM)
                layer.values = torch.zeros(1, HEADS, length, HEAD_DIM)
                # Les tokens du préfixe sont mémorisés dans le cache : c'est
                # ce qui permet au test de vérifier que chaque branche reçoit
                # **le** préfixe demandé, et non celui d'un job antérieur.
                layer.values[0, :, :, 0] = tokens.unsqueeze(0)
                layers.append(layer)
            return _FakeOutput(logits, _FakeCache(layers))
        self.seen_prefixes.append(
            [float(value) for value in past_key_values.layers[0].values[0, 0, :, 0]]
        )
        return _FakeOutput(logits, past_key_values)


@pytest.fixture()
def scorer() -> Qwen3TeacherScorer:
    return Qwen3TeacherScorer(_FakeModel(), device="cpu")


def test_scorer_implements_the_j13_contract(scorer: Qwen3TeacherScorer) -> None:
    """C1 : le scorer hérite du contrat J13 et expose le vocabulaire."""
    assert isinstance(scorer, TeacherScorer)
    assert scorer.vocab_size == VOCAB_SIZE


def test_teacher_is_frozen(scorer: Qwen3TeacherScorer) -> None:
    """C9 : le professeur est gelé (eval + requires_grad=False)."""
    assert scorer.model.training is False
    assert all(not parameter.requires_grad for parameter in scorer.model.parameters())


def test_causal_prior_is_a_distribution(scorer: Qwen3TeacherScorer) -> None:
    """C4 : le prior somme à 1 et ne contient ni NaN ni inf."""
    prior = scorer.causal_prior((1, 2, 3))
    assert prior.shape == (VOCAB_SIZE,)
    assert np.all(np.isfinite(prior))
    assert np.all(prior >= 0.0)
    assert abs(float(prior.sum()) - 1.0) <= 1e-4


def test_cache_path_matches_replay_path(scorer: Qwen3TeacherScorer) -> None:
    """C2 (logique) : le chemin cache et le chemin de rejeu coïncident.

    Sur le modèle factice, les deux chemins sont **identiques par
    construction** : c'est ce qui rend le test du vrai modèle interpretable.
    """
    prefix = (1, 2, 3)
    candidates = (4, 5, 6)
    suffix = (7, 8)

    prior = scorer.causal_prior(prefix)
    cached = scorer.suffix_loglikelihood(prefix, candidates, suffix)
    replay_prior, replayed = scorer.replay_job(prefix, candidates, suffix)

    assert cached.shape == replayed.shape == (3,)
    assert np.max(np.abs(cached - replayed)) <= 1e-6
    assert np.max(np.abs(prior - replay_prior)) <= 1e-6


def test_suffix_scores_depend_on_the_candidate(scorer: Qwen3TeacherScorer) -> None:
    """Un score de suffixe identique pour tous les candidats serait faux."""
    prefix = (1, 2, 3)
    suffix = (7, 8)
    scores = scorer.suffix_loglikelihood(prefix, (4, 5, 6), suffix)
    assert len(set(scores.tolist())) > 1


def test_prefix_is_prefilled_once_per_job(scorer: Qwen3TeacherScorer) -> None:
    """Le préfixe commun n'est prérempli qu'une fois pour un job complet.

    `score()` appelle le prior puis les scores : sans mémoire de préfixe, le
    préfixe serait prérempli deux fois (deux fois le coût de J14).
    """
    scorer.model.calls.clear()
    request = ScoringRequest(
        prefix_ids=(1, 2, 3),
        suffix_ids=(7, 8),
        candidate_ids=(4, 5),
        reference_id=9,
    )
    scorer.score(request)
    # Deux forwards : le préremplissage du préfixe (3 tokens) et le forward des
    # branches (candidat + suffixe = 3 tokens).
    assert scorer.model.calls == [3, 3]


def test_score_request_returns_a_valid_result(scorer: Qwen3TeacherScorer) -> None:
    """C1 : `score(ScoringRequest)` rend un `ScoringResult` valide."""
    request = ScoringRequest(
        prefix_ids=(1, 2, 3),
        suffix_ids=(7, 8),
        candidate_ids=(4, 5),
        reference_id=9,
    )
    result = scorer.score(request)
    assert result.candidate_ids == (4, 5)
    assert result.candidate_loglikelihoods.shape == (2,)
    assert np.isfinite(result.reference_loglikelihood)
    assert abs(float(result.prior.sum()) - 1.0) <= 1e-4


def test_reference_token_is_not_duplicated(scorer: Qwen3TeacherScorer) -> None:
    """La référence déjà présente dans les candidats n'est pas dupliquée."""
    request = ScoringRequest(
        prefix_ids=(1, 2, 3),
        suffix_ids=(7, 8),
        candidate_ids=(4, 9),
        reference_id=9,
    )
    result = scorer.score(request)
    assert result.candidate_ids == (4, 9)
    assert result.candidate_loglikelihoods.shape == (2,)


def test_prefix_cache_does_not_leak_between_jobs(scorer: Qwen3TeacherScorer) -> None:
    """Un job ne doit pas hériter du cache du job précédent.

    Le cache Hugging Face est mutable : le réutiliser sans le reconstruire
    fausserait les jobs suivants. Le test vérifie que **chaque branche**
    reçoit exactement le préfixe demandé, et que le résultat est
    reproductible d'un job à l'autre.
    """
    scorer.suffix_loglikelihood((1, 2, 3), (4,), (7, 8))
    scorer.suffix_loglikelihood((2, 3, 4), (4,), (7, 8))
    scorer.suffix_loglikelihood((1, 2, 3), (4,), (7, 8))
    assert scorer.model.seen_prefixes == [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [1.0, 2.0, 3.0]]


def test_same_prefix_gives_reproducible_scores(scorer: Qwen3TeacherScorer) -> None:
    """Un même préfixe redonne exactement les mêmes scores."""
    first = scorer.suffix_loglikelihood((1, 2, 3), (4, 5), (7, 8))
    second = scorer.suffix_loglikelihood((1, 2, 3), (4, 5), (7, 8))
    assert np.array_equal(first, second)


def test_empty_prefix_is_rejected(scorer: Qwen3TeacherScorer) -> None:
    """Un préfixe vide est refusé : le prior causal exige un token."""
    with pytest.raises(ValueError, match="prefix_ids"):
        scorer.causal_prior(())


def test_empty_candidates_are_rejected(scorer: Qwen3TeacherScorer) -> None:
    """Une liste de candidats vide est refusée."""
    with pytest.raises(ValueError, match="candidate_ids"):
        scorer.suffix_loglikelihood((1, 2), (), (3,))


def test_reset_prefix_cache_forces_a_new_prefill(scorer: Qwen3TeacherScorer) -> None:
    """`reset_prefix_cache` invalide la mémoire de préfixe."""
    scorer.model.calls.clear()
    scorer.causal_prior((1, 2, 3))
    scorer.causal_prior((1, 2, 3))
    assert len(scorer.model.calls) == 1
    scorer.reset_prefix_cache()
    scorer.causal_prior((1, 2, 3))
    assert len(scorer.model.calls) == 2


def test_teacher_topk_is_deterministic_and_sorted() -> None:
    """Le top-k est trié par identifiant croissant et déterministe."""
    prior = np.array([0.1, 0.5, 0.2, 0.2], dtype=np.float64)
    first = teacher_topk(prior, 3)
    second = teacher_topk(prior, 3)
    assert first.tolist() == [1, 2, 3]
    assert np.array_equal(first, second)


def test_teacher_topk_rejects_invalid_k() -> None:
    """`k` doit être >= 1."""
    prior = np.array([0.5, 0.5], dtype=np.float64)
    with pytest.raises(ValueError, match="k"):
        teacher_topk(prior, 0)
