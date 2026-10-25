"""Tests du `TeacherScorer` réel d-OPD (J17) — harnais, torch optionnel.

Ces tests n'exercent que la **logique** du scorer sur un modèle factice : ils
ne chargent aucun checkpoint, n'utilisent aucun GPU et ne téléchargent rien.
Le contrat J13 reste vérifié par `test_dopd_interfaces.py` (paquet pur).

Les tests sont ignorés si `torch` n'est pas installé (`pytest.importorskip`) :
la collecte doit passer sans torch (critère C13).

    pytest modules/dopd/tests/test_dopd_harness_teacher_scorer.py -q
"""

from __future__ import annotations

import math
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
    """Modèle déterministe **sensible au préfixe, aux positions et au masque**.

    Le modèle reproduit les propriétés essentielles du contrat :

    * le prior causal est une distribution sur le vocabulaire ;
    * le score de suffixe dépend **du candidat** (sinon le contrôle
      d'exactitude ne testerait rien) ;
    * le score dépend **du préfixe** : deux préfixes différents donnent des
      scores différents (sans cela, une implémentation qui ignore le préfixe
      passerait tous les tests) ;
    * le score dépend de la **position** du token : une lecture de logit
      décalée d'une position change le résultat ;
    * le chemin cache et le chemin de rejeu donnent le **même** résultat
      logique — c'est ce que le contrôle d'exactitude vérifie sur le vrai
      modèle ;
    * le cache prérempli contient une clé et une valeur par couche.

    La dépendance au préfixe et à la position est obtenue en propageant, dans
    le « contexte », la somme pondérée des tokens du préfixe **et** la
    position absolue de chaque token. Le chemin cache reçoit le préfixe via
    le cache ; le chemin de rejeu le reçoit via la séquence complète. Les
    deux reconstructs donc le même contexte, ce qui rend leur comparaison
    significative.
    """

    def __init__(self) -> None:
        self.config = _FakeConfig()
        self.training = True
        self.calls: list[int] = []
        self.seen_prefixes: list[list[float]] = []
        self.seen_position_ids: list[list[int]] = []
        self.seen_attention_masks: list[list[int]] = []
        self._parameter = torch.nn.Parameter(torch.zeros(1), requires_grad=True)

    def parameters(self):
        return iter((self._parameter,))

    def eval(self) -> None:
        self.training = False

    def to(self, device):  # pragma: no cover - interface de compatibilité
        return self

    @staticmethod
    def _aggregate(
        tokens: torch.Tensor,
        positions: torch.Tensor,
        visible: torch.Tensor,
    ) -> torch.Tensor:
        """Agrégat causal par position : somme des tokens **antérieurs ou égaux**.

        Chaque token `x` à la position absolue `p` contribue `x * (p + 1)`.
        L'agrégat à la position `q` est la somme des contributions des
        positions `0..q` : c'est l'équivalent d'une attention causale
        uniforme sur tout le contexte visible.

        Seuls les tokens **visibles** (`visible = 1`) contribuent : un token
        masqué est exclu du contexte, exactement comme le fait une attention
        masquée. C'est ce qui rend le fake sensible aux **valeurs** du masque
        et non seulement à sa présence.
        """
        contributions = (
            tokens.to(torch.float32)
            * (positions.to(torch.float32) + 1.0)
            * visible.to(torch.float32)
        )
        return contributions.cumsum(dim=-1)

    def _logits(
        self,
        input_ids: torch.Tensor,
        aggregates: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Logits déterministes, dépendants de l'agrégat et de la position.

        Le logit du vocabulaire `v` au token de position `p` vaut
        `cos(sin(token * 0.37) + v + aggregate * 0.01 + p * 0.11)`. L'agrégat
        vient du contexte visible (préfixe + tokens antérieurs de la
        branche), la position de l'index absolu : un décalage de position ou
        un préfixe ignoré change le résultat.
        """
        token = input_ids.to(torch.float32)
        base = torch.sin(token * 0.37)
        shift = aggregates * 0.01 + position_ids.to(torch.float32) * 0.11
        return torch.stack(
            [torch.cos(base + shift + offset) for offset in range(VOCAB_SIZE)],
            dim=-1,
        )

    def __call__(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: object | None = None,
        use_cache: bool = True,
        logits_to_keep: int = 0,
    ) -> _FakeOutput:
        self.calls.append(int(input_ids.shape[-1]))
        if position_ids is None:
            position_ids = torch.arange(
                input_ids.shape[-1], dtype=torch.long, device=input_ids.device
            ).unsqueeze(0).expand(input_ids.shape[0], -1)
        if position_ids is not None:
            self.seen_position_ids.append(
                [int(value) for value in position_ids[0].tolist()]
            )
        if attention_mask is not None:
            self.seen_attention_masks.append(
                [int(value) for value in attention_mask[0].tolist()]
            )
        # Visibilité : le masque vaut 1 pour un token visible, 0 pour un
        # token masqué. Sans masque, tout est visible.
        if attention_mask is None:
            visible = torch.ones_like(input_ids, dtype=torch.long)
        else:
            visible = attention_mask.to(torch.long)
        if past_key_values is None:
            # Préremplissage (batch 1) ou rejeu intégral (batch = branches) :
            # le contexte visible est la séquence entière, l'agrégat est
            # causal et calculé **par ligne**.
            aggregates = self._aggregate(input_ids, position_ids, visible)
            logits = self._logits(input_ids, aggregates, position_ids)
            if logits_to_keep:
                logits = logits[:, -logits_to_keep:, :]
            if input_ids.shape[0] == 1:
                layers = []
                tokens = input_ids[0].to(torch.float32)
                for _ in range(LAYERS):
                    layer = _FakeCacheLayer()
                    length = int(input_ids.shape[-1])
                    layer.keys = torch.zeros(1, HEADS, length, HEAD_DIM)
                    layer.values = torch.zeros(1, HEADS, length, HEAD_DIM)
                    # Les tokens du préfixe sont mémorisés dans le cache :
                    # c'est ce qui permet au test de vérifier que chaque
                    # branche reçoit **le** préfixe demandé, et non celui
                    # d'un job antérieur.
                    layer.values[0, :, :, 0] = tokens.unsqueeze(0)
                    layers.append(layer)
                cache = _FakeCache(layers)
                return _FakeOutput(logits, cache)
            # Rejeu intégral : pas de cache.
            return _FakeOutput(logits, None)
        # Branches : le contexte visible est reconstitué depuis le **cache**
        # (le préfixe) et les tokens de la branche — exactement comme le
        # vrai modèle, où l'attention de la branche porte sur le préfixe
        # mis en cache plus les tokens qui la précèdent.
        cached = past_key_values.layers[0].values[0, 0, :, 0]
        self.seen_prefixes.append([float(value) for value in cached.tolist()])
        prefix_length = int(cached.shape[-1])
        prefix_positions = torch.arange(
            prefix_length, dtype=torch.long, device=input_ids.device
        )
        # Le masque reçu couvre le préfixe **et** les tokens de la branche.
        # Dans un vrai modèle à cache, ce sont les `prefix_length` premières
        # valeurs du masque **de la branche** qui commandent la visibilité
        # des clés déjà en cache : le fake applique la même règle.
        prefix_visible = visible[:, :prefix_length]
        branch_visible = visible[:, prefix_length:]
        # Contributions par ligne de lot : (batch, longueur).
        prefix_contribution = (
            cached.to(torch.float32).unsqueeze(0)
            * (prefix_positions.to(torch.float32) + 1.0).unsqueeze(0)
        ) * prefix_visible.to(torch.float32)
        branch_contribution = input_ids.to(torch.float32) * (
            position_ids.to(torch.float32) + 1.0
        ) * branch_visible.to(torch.float32)
        # Agrégat cumulé : préfixe puis branche.
        aggregates = torch.cumsum(
            torch.cat((prefix_contribution, branch_contribution), dim=-1),
            dim=-1,
        )[:, prefix_length:]
        logits = self._logits(input_ids, aggregates, position_ids)
        if logits_to_keep:
            logits = logits[:, -logits_to_keep:, :]
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


def _expected_suffix_score(
    prefix: tuple[int, ...],
    candidate: int,
    suffix: tuple[int, ...],
) -> float:
    """Score de suffixe attendu, recalculé **hors du scorer**.

    Le modèle factice produit, pour un token `t` à la position absolue `p`,
    les logits `logits[v] = cos(sin(t * 0.37) + v + agg(p) * 0.01 + p * 0.11)`,
    où `agg(p)` est la somme pondérée `Σ x_q * (q + 1)` des tokens `x_q` aux
    positions `q <= p` (agrégat causal uniforme).

    Le contexte de la branche est donc : le préfixe aux positions `0..P-1`,
    le candidat à la position `P`, puis les tokens du suffixe à `P+1..P+S`.
    Le score est la somme des log-probabilités du suffixe.

    Ce calcul ne passe par **aucune** méthode du scorer : il sert de vérité
    absolue, là où `test_cache_path_matches_replay_path` ne prouve que la
    *coïncidence* des deux chemins (les deux pourraient être faux ensemble).
    """
    prefix_length = len(prefix)
    # Contributions : (token, position) dans l'ordre causal.
    contributions = [
        (token, position) for position, token in enumerate(prefix)
    ]
    contributions.append((candidate, prefix_length))
    for offset, token in enumerate(suffix):
        contributions.append((token, prefix_length + 1 + offset))

    # Agrégat causal au début de chaque token de suffixe.
    totals: list[float] = []
    running = 0.0
    for token, position in contributions:
        running += token * (position + 1)
        totals.append(running)

    total = 0.0
    branch = [candidate, *suffix]
    for offset, token in enumerate(suffix):
        # Le logit qui prédit `suffix[offset]` est celui du token **à la
        # position qui la précède dans la branche** : le candidat (position
        # P) prédit `suffix[0]`, `suffix[offset-1]` (position P+offset)
        # prédit `suffix[offset]`.
        predictor_position = prefix_length + offset
        predictor_token = branch[offset]
        aggregate = totals[predictor_position]
        base = math.sin(predictor_token * 0.37)
        shift = aggregate * 0.01 + predictor_position * 0.11
        logits = np.array(
            [math.cos(base + shift + o) for o in range(VOCAB_SIZE)], dtype=np.float64
        )
        shifted = logits - logits.max()
        log_probs = shifted - math.log(float(np.exp(shifted).sum()))
        total += float(log_probs[token])
    return total


def test_suffix_scores_match_an_independent_recomputation(
    scorer: Qwen3TeacherScorer,
) -> None:
    """Les scores doivent égaler une valeur **absolue**, pas seulement coïncider.

    Un décalage d'une position dans la lecture des logits donnerait deux
    chemins (cache et rejeu) d'accord entre eux mais faux tous les deux : ce
    test est le seul à le détecter.
    """
    prefix = (1, 2, 3)
    suffix = (7, 8)
    candidates = (4, 5, 6)

    cached = scorer.suffix_loglikelihood(prefix, candidates, suffix)
    _, replayed = scorer.replay_job(prefix, candidates, suffix)
    expected = np.array(
        [
            _expected_suffix_score(prefix, candidate, suffix)
            for candidate in candidates
        ],
        dtype=np.float64,
    )

    assert np.max(np.abs(cached - expected)) <= 1e-6
    assert np.max(np.abs(replayed - expected)) <= 1e-6


def test_prior_matches_an_independent_recomputation(scorer: Qwen3TeacherScorer) -> None:
    """Le prior causal doit égaler le softmax des logits du dernier préfixe.

    Contrôle la **position** de lecture du logit : un logit lu au mauvais
    endroit donnerait un prior encore normalisé, mais faux.
    """
    prefix = (1, 2, 3)
    position = len(prefix) - 1
    aggregate = sum(token * (index + 1) for index, token in enumerate(prefix))
    base = math.sin(prefix[-1] * 0.37)
    shift = aggregate * 0.01 + position * 0.11
    logits = np.array(
        [math.cos(base + shift + offset) for offset in range(VOCAB_SIZE)],
        dtype=np.float64,
    )
    shifted = logits - logits.max()
    expected = np.exp(shifted) / float(np.exp(shifted).sum())

    prior = scorer.causal_prior(prefix)
    # Le scorer calcule le softmax en float32 (§3.6) ; l'attendu est en
    # float64. La tolérance couvre l'arrondi float32 (~1e-8 sur des
    # probabilités de l'ordre de 0,1), pas une erreur de lecture de logit
    # (qui serait de l'ordre de 0,1).
    assert np.max(np.abs(prior - expected)) <= 1e-6


def test_empty_suffix_yields_zero_scores(scorer: Qwen3TeacherScorer) -> None:
    """Sans futur visible, le score de suffixe est vide (zéros)."""
    scores = scorer.suffix_loglikelihood((1, 2, 3), (4, 5), ())
    assert scores.shape == (2,)
    assert np.all(scores == 0.0)


def test_sliding_window_models_are_refused_explicitly(
    scorer: Qwen3TeacherScorer,
) -> None:
    """Un modèle à attention glissante doit être **refusé nommément**.

    Le cache de préfixe reconstruit un `DynamicCache` sans configuration : il
    suppose une attention pleine. Sans garde-fou, l'erreur remonte du noyau
    d'attention sous une forme cryptique ; le refus doit nommer la cause.
    """
    scorer.model.config.layer_types = ["full_attention", "sliding_attention"]
    with pytest.raises(ValueError, match="glissante"):
        scorer.suffix_loglikelihood((1, 2, 3), (4, 5), (7, 8))


def test_sliding_window_config_is_refused_explicitly(
    scorer: Qwen3TeacherScorer,
) -> None:
    """Un `sliding_window` déclaré doit être refusé, même sans `layer_types`."""
    scorer.model.config.sliding_window = 8
    with pytest.raises(ValueError, match="sliding_window"):
        scorer.suffix_loglikelihood((1, 2, 3), (4, 5), (7, 8))


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


# ---------------------------------------------------------------------------
# Classe de bug « masque / positions / conditionnement par le préfixe »
#
# Un fake dont les logits ne dépendent que du token laisse passer une
# implémentation qui (i) ignore le préfixe, (ii) passe un masque décalé ou
# (iii) lit les logits à la mauvaise position. Les tests ci-dessous ferment
# cette hole : le fake est sensible au préfixe, aux positions et au masque,
# et chaque propriété est vérifiée **explicitement**.
# ---------------------------------------------------------------------------


def test_scores_change_when_the_prefix_changes(scorer: Qwen3TeacherScorer) -> None:
    """Deux préfixes différents doivent donner des scores différents.

    C'est le test qui détecte une implémentation ignorant le préfixe : sans
    cette propriété, tous les autres contrôles (cache vs rejeu) peuvent
    passer alors que le modèle ne conditionne pas sur le contexte.
    """
    candidates = (4, 5)
    suffix = (7, 8)
    first = scorer.suffix_loglikelihood((1, 2, 3), candidates, suffix)
    second = scorer.suffix_loglikelihood((4, 5, 6), candidates, suffix)
    assert not np.allclose(first, second)


def test_the_model_receives_explicit_positions_and_mask(
    scorer: Qwen3TeacherScorer,
) -> None:
    """Le préremplissage et les branches portent une géométrie explicite.

    Préremplissage : positions `0..P-1`, masque plein sur `P` tokens.
    Branches : positions `P..P+S` (le candidat à `P`), masque plein sur
    `P + 1 + S` tokens (préfixe + candidat + suffixe).

    Une géométrie absente ou décalée est ici **visible**, pas silencieuse.
    """
    prefix = (1, 2, 3)
    suffix = (7, 8)
    scorer.suffix_loglikelihood(prefix, (4, 5), suffix)
    # Préremplissage puis forward des branches.
    assert scorer.model.seen_position_ids == [[0, 1, 2], [3, 4, 5]]
    assert scorer.model.seen_attention_masks == [
        [1, 1, 1],
        [1, 1, 1, 1, 1, 1],
    ]


def test_a_shifted_position_would_change_the_scores() -> None:
    """Le score dépend effectivement de la position : garde-fou du décalage.

    Si le fake était insensible aux positions, un off-by-one de `position_ids`
    passerait tous les tests. Ce test **vérifie la sensibilité** elle-même.
    """
    prefix = (1, 2, 3)
    candidates = (4, 5)
    suffix = (7, 8)

    reference = Qwen3TeacherScorer(_FakeModel(), device="cpu")
    expected = reference.suffix_loglikelihood(prefix, candidates, suffix)

    class _ShiftedPositionsModel(_FakeModel):
        """Fake dont les positions sont décalées d'une unité."""

        def __call__(self, input_ids, **kwargs):
            position_ids = kwargs.get("position_ids")
            if position_ids is not None:
                kwargs["position_ids"] = position_ids + 1
            return super().__call__(input_ids, **kwargs)

    shifted = Qwen3TeacherScorer(_ShiftedPositionsModel(), device="cpu")
    actual = shifted.suffix_loglikelihood(prefix, candidates, suffix)
    assert not np.allclose(expected, actual)


def test_chunked_scoring_matches_unchunked_scoring() -> None:
    """Le découpage en lots ne change pas le résultat.

    Le plafond `max_prefix_cache_bytes` (§3.6) peut imposer un découpage des
    branches en plusieurs forwards. Le résultat doit être **exactement** le
    même qu'en un seul forward : c'est la garantie que le découpage est une
    optimisation sans effet numérique.
    """
    model = _FakeModel()
    prefix = (1, 2, 3)
    candidates = (4, 5, 6, 7)
    suffix = (8, 9)

    unchunked = Qwen3TeacherScorer(model, device="cpu")
    expected = unchunked.suffix_loglikelihood(prefix, candidates, suffix)

    # Plafond assez petit pour forcer un lot de 1 branche.
    chunked_model = _FakeModel()
    chunked = Qwen3TeacherScorer(chunked_model, device="cpu")
    chunked.max_prefix_cache_bytes = 1
    actual = chunked.suffix_loglikelihood(prefix, candidates, suffix)

    assert actual.shape == expected.shape
    assert np.max(np.abs(actual - expected)) <= 1e-6
    # Le découpage a bien eu lieu : plusieurs forwards de branches.
    assert len(chunked_model.calls) > len(unchunked.model.calls)


def test_empty_suffix_does_not_prefill(scorer: Qwen3TeacherScorer) -> None:
    """Un suffixe vide ne déclenche aucun forward de préremplissage.

    `log p(∅ | …) = 0` par convention : le préremplissage du préfixe serait
    un forward intégral inutile.
    """
    scorer.model.calls.clear()
    scores = scorer.suffix_loglikelihood((1, 2, 3), (4, 5), ())
    assert scores.shape == (2,)
    assert np.all(scores == 0.0)
    assert scorer.model.calls == []


def test_the_fake_is_sensitive_to_the_mask_values() -> None:
    """Le fake doit dépendre des **valeurs** du masque, pas de sa présence.

    Sans cette propriété, un fake qui enregistre le masque sans l'utiliser
    laisserait passer une implémentation qui passe un masque faux : la
    couverture de la classe de bug « masque » serait alors **illusoire**.
    Ce test verrouille la sensibilité elle-même.
    """
    prefix = (1, 2, 3)
    candidates = (4, 5)
    suffix = (7, 8)

    class _MaskedModel(_FakeModel):
        """Fake dont tous les tokens sont masqués."""

        def __call__(self, input_ids, **kwargs):
            attention_mask = kwargs.get("attention_mask")
            if attention_mask is not None:
                kwargs["attention_mask"] = torch.zeros_like(attention_mask)
            return super().__call__(input_ids, **kwargs)

    reference = Qwen3TeacherScorer(_FakeModel(), device="cpu")
    expected = reference.suffix_loglikelihood(prefix, candidates, suffix)

    masked = Qwen3TeacherScorer(_MaskedModel(), device="cpu")
    actual = masked.suffix_loglikelihood(prefix, candidates, suffix)

    assert not np.allclose(expected, actual)


def test_unsupported_transformers_version_is_refused_explicitly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Une version de `transformers` trop ancienne est refusée nommément.

    L'API cache `DynamicCache(ddp_cache_data=...)` n'existe pas avant 4.56.2 :
    sans garde-fou, l'erreur remonterait au milieu du scoring, sous une
    forme obscure.
    """
    import teacher_scorer_qwen3 as module

    monkeypatch.setattr(module.transformers, "__version__", "4.46.2")
    with pytest.raises(RuntimeError, match="trop ancien"):
        module.assert_supported_transformers()


# ---------------------------------------------------------------------------
# Contrôle sur **modèle réel**
#
# Le fake vérifie la logique du scorer ; seul un vrai `Qwen3ForCausalLM`
# vérifie que le cache de préfixe, les positions et le masque se comportent
# comme le suppose l'implémentation. La configuration est minuscule (2
# couches, 4 têtes, 32 embeddings) : le test reste instantané sur CPU et ne
# télécharge rien.
# ---------------------------------------------------------------------------


def _tiny_qwen3():
    """Construit un `Qwen3ForCausalLM` minuscule, **sans téléchargement**."""
    pytest.importorskip("transformers")
    qwen3 = pytest.importorskip("transformers.models.qwen3")
    torch.manual_seed(0)
    config = qwen3.Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        attn_implementation="eager",
    )
    model = qwen3.Qwen3ForCausalLM(config)
    model.eval()
    return model


def test_real_model_cache_path_matches_full_replay() -> None:
    """Sur un vrai Qwen3, le chemin cache égale le rejeu intégral.

    Ce test est le contrôle qui manque à un fake : il détecte un masque
    décalé, un `position_ids` faux ou une lecture de logit à la mauvaise
    position — des fautes que le fake, par construction, ne voit pas.
    """
    model = _tiny_qwen3()
    scorer = Qwen3TeacherScorer(model, device="cpu")
    prefix = (3, 11, 7, 5, 2)
    candidates = (9, 17, 23)
    suffix = (4, 8)

    cached = scorer.suffix_loglikelihood(prefix, candidates, suffix)
    replay_prior, replayed = scorer.replay_job(prefix, candidates, suffix)
    prior = scorer.causal_prior(prefix)

    assert cached.shape == replayed.shape == (3,)
    assert np.max(np.abs(cached - replayed)) <= 1e-4
    assert np.max(np.abs(prior - replay_prior)) <= 1e-4


def test_real_model_scores_depend_on_the_prefix() -> None:
    """Sur un vrai Qwen3, changer le préfixe change les scores.

    Preuve que le conditionnement par le contexte est effectif : une
    implémentation qui ignorerait le préfixe serait ici attrapée.
    """
    model = _tiny_qwen3()
    scorer = Qwen3TeacherScorer(model, device="cpu")
    candidates = (9, 17)
    suffix = (4, 8)
    first = scorer.suffix_loglikelihood((3, 11, 7), candidates, suffix)
    second = scorer.suffix_loglikelihood((21, 2, 13), candidates, suffix)
    assert not np.allclose(first, second)


def test_real_model_chunked_matches_unchunked() -> None:
    """Sur un vrai Qwen3, le découpage en lots ne change rien."""
    model = _tiny_qwen3()
    prefix = (3, 11, 7, 5)
    candidates = (9, 17, 23, 31)
    suffix = (4, 8)

    unchunked = Qwen3TeacherScorer(model, device="cpu")
    expected = unchunked.suffix_loglikelihood(prefix, candidates, suffix)

    chunked = Qwen3TeacherScorer(model, device="cpu")
    chunked.max_prefix_cache_bytes = 1
    actual = chunked.suffix_loglikelihood(prefix, candidates, suffix)

    assert np.max(np.abs(actual - expected)) <= 1e-4
