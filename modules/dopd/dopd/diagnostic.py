"""Noyau NumPy pur du diagnostic Annexe D de d-OPD.

Ce module contient uniquement les calculs déterministes du diagnostic
énumerable : support teacher restreint, loi jointe, posterior exact, cibles
causal/future-aware et KL. Il ne dépend que de NumPy et de la bibliothèque
standard ; le chargement de Qwen et les forwards torch vivent dans
``modules/dopd/harness/diagnostic_annexe_d.py`` (ADR-001).

Le protocole J14 et ses critères pré-enregistrés sont dans
``modules/dopd/docs/DOPD_SPEC.md`` §6.9.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import math
import numpy as np

__all__ = [
    "GateDecision",
    "RestrictedSupport",
    "bootstrap_delta_ci",
    "evaluate_gate",
    "exact_token_posteriors",
    "exact_token_posteriors_reference",
    "kl_exact_to_target",
    "sample_compatible_index",
    "summarize_marginals",
    "verify_enumeration_agreement",
    "bootstrap_prompt_ci",
    "evaluate_multiprompt",
    "per_prompt_marginals",
    "sign_test",
    "truncate_causal_gap",
]


@dataclass(frozen=True)
class RestrictedSupport:
    """Support fini construit position par position comme l'Annexe D.1.

    ``supports[d]`` contient les identifiants de tokens retenus à la position
    ``d``. Pour chaque profondeur ``d``, ``prefixes[d]`` énumère tous les
    préfixes compatibles de longueur ``d`` et ``conditional_log_probs[d]``
    contient, pour chaque préfixe, les log-probabilités teacher correspondant
    aux tokens de ``supports[d]``.

    Ces objets sont construits par le harnais à partir des sorties du modèle,
    mais leur manipulation et les énumérations restent ici en NumPy pur.
    """

    supports: tuple[np.ndarray, ...]
    prefixes: tuple[tuple[tuple[int, ...], ...], ...]
    conditional_log_probs: tuple[np.ndarray, ...]

    def __post_init__(self) -> None:
        depth = len(self.supports)
        if depth == 0:
            raise ValueError("le support doit contenir au moins une position")
        if len(self.prefixes) != depth or len(self.conditional_log_probs) != depth:
            raise ValueError("supports, prefixes et conditional_log_probs doivent être alignés")
        expected_prefixes = 1
        for position, support in enumerate(self.supports):
            ids = np.asarray(support, dtype=np.int64)
            if ids.ndim != 1 or ids.size == 0:
                raise ValueError(f"support[{position}] doit être un vecteur non vide")
            if np.unique(ids).size != ids.size:
                raise ValueError(f"support[{position}] contient des doublons")
            if len(self.prefixes[position]) != expected_prefixes:
                raise ValueError(
                    f"prefixes[{position}] a {len(self.prefixes[position])} éléments, "
                    f"attendu {expected_prefixes}"
                )
            conditional = np.asarray(self.conditional_log_probs[position], dtype=np.float64)
            if conditional.shape != (expected_prefixes, ids.size):
                raise ValueError(
                    f"conditional_log_probs[{position}] a la forme {conditional.shape}, "
                    f"attendu {(expected_prefixes, ids.size)}"
                )
            expected_prefixes *= ids.size

    @property
    def block_size(self) -> int:
        """Nombre de positions du bloc."""
        return len(self.supports)

    def _prefix_lookup(self, position: int) -> dict[tuple[int, ...], int]:
        """Table de correspondance prefixe -> index, mise en cache par position."""
        cache = self.__dict__.get("_prefix_lookup_cache")
        if cache is None:
            cache = {}
            object.__setattr__(self, "_prefix_lookup_cache", cache)
        lookup = cache.get(position)
        if lookup is None:
            lookup = {
                prefix: index for index, prefix in enumerate(self.prefixes[position])
            }
            cache[position] = lookup
        return lookup

    def prefix_index(self, position: int, prefix: tuple[int, ...]) -> int:
        """Index d'un préfixe compatible à la profondeur ``position``."""
        if len(prefix) != position:
            raise ValueError(f"préfixe de longueur {len(prefix)}, attendu {position}")
        lookup = self._prefix_lookup(position)
        try:
            return lookup[prefix]
        except KeyError as exc:
            raise ValueError(
                f"préfixe {prefix!r} absent du support à la position {position}"
            ) from exc

    def enumerate_joint(self) -> tuple[np.ndarray, np.ndarray]:
        """Énumère toutes les séquences complètes et leur log-probabilité jointe.

        La loi est renormalisée sur le support fini, conformément à l'équation
        22 de l'Annexe D.
        """
        sequences: list[tuple[int, ...]] = [()]
        log_weights: list[float] = [0.0]
        for position in range(self.block_size):
            next_sequences: list[tuple[int, ...]] = []
            next_weights: list[float] = []
            conditional = self.conditional_log_probs[position]
            for prefix_index, (prefix, prefix_weight) in enumerate(
                zip(self.prefixes[position], log_weights, strict=True)
            ):
                for candidate_index, token_id in enumerate(self.supports[position]):
                    next_sequences.append(prefix + (int(token_id),))
                    next_weights.append(
                        prefix_weight + float(conditional[prefix_index, candidate_index])
                    )
            sequences = next_sequences
            log_weights = next_weights
        sequence_array = np.asarray(sequences, dtype=np.int64)
        log_joint = np.asarray(log_weights, dtype=np.float64)
        log_joint -= _logsumexp(log_joint)
        return sequence_array, log_joint

    def causal_prior(self, prefix: tuple[int, ...], position: int) -> np.ndarray:
        """Prior causal restreint à la position ``position``.

        Le préfixe conditioning doit provenir du support fini. Le résultat est
        renormalisé sur ``supports[position]`` ; il s'agit donc de la cible
        causale de l'équation 25, exprimée dans le support de la position.
        """
        if position < 0 or position >= self.block_size:
            raise ValueError(f"position {position} hors de [0, {self.block_size})")
        prefix_index = self.prefix_index(position, prefix)
        logs = self.conditional_log_probs[position][prefix_index]
        return _normalized_from_logs(logs)


@dataclass(frozen=True)
class _Compatibility:
    mask: np.ndarray
    weights: np.ndarray


def _compatibility(
    sequences: np.ndarray,
    log_joint: np.ndarray,
    visible: dict[int, int],
) -> _Compatibility:
    compatible = np.ones(sequences.shape[0], dtype=bool)
    for position, token_id in visible.items():
        compatible &= sequences[:, position] == int(token_id)
    weights = np.zeros_like(log_joint)
    if not np.any(compatible):
        raise ValueError("aucune séquence compatible avec l'état visible")
    shifted = log_joint[compatible] - np.max(log_joint[compatible])
    weights[compatible] = np.exp(shifted)
    weights[compatible] /= np.sum(weights[compatible])
    return _Compatibility(mask=compatible, weights=weights)


def exact_token_posteriors(
    sequences: np.ndarray,
    log_joint: np.ndarray,
    visible: dict[int, int],
    supports: tuple[np.ndarray, ...],
) -> dict[int, np.ndarray]:
    """Posterior token exact conditionné sur l'état visible (Annexe D.2).

    Pour chaque position masquée, la fonction somme les probabilités jointe
    teacher de toutes les séquences compatibles dont le token à cette position
    est fixé. C'est l'équation 8 calculée par énumération exhaustive.
    """
    compatibility = _compatibility(sequences, log_joint, visible)
    result: dict[int, np.ndarray] = {}
    for position, support in enumerate(supports):
        if position in visible:
            continue
        marginal = np.zeros(support.size, dtype=np.float64)
        lookup = {int(token_id): index for index, token_id in enumerate(support)}
        indices = np.fromiter(
            (lookup.get(int(token), -1) for token in sequences[:, position]),
            dtype=np.int64,
            count=sequences.shape[0],
        )
        selected = compatibility.mask & (indices >= 0)
        if not np.any(selected):
            raise ValueError(f"aucune séquence compatible à la position {position}")
        marginal = np.bincount(
            indices[selected],
            weights=compatibility.weights[selected],
            minlength=support.size,
        ).astype(np.float64)
        total = float(np.sum(marginal))
        if total <= 0.0 or not np.isfinite(total):
            raise ValueError(f"marginale exacte invalide à la position {position}")
        result[position] = marginal / total
    return result


def exact_token_posteriors_reference(
    sequences: np.ndarray,
    log_joint: np.ndarray,
    visible: dict[int, int],
    supports: tuple[np.ndarray, ...],
) -> dict[int, np.ndarray]:
    """Implémentation de contrôle, explicite et indépendante de la version vectorisée.

    Elle sert au test d'intégrité numérique exigé par la porte J14 : chaque
    marginale est recalculée en parcourant toutes les séquences une par une.
    """
    compatible_indices = [
        index
        for index, sequence in enumerate(sequences)
        if all(int(sequence[position]) == token_id for position, token_id in visible.items())
    ]
    if not compatible_indices:
        raise ValueError("aucune séquence compatible avec l'état visible")
    maximum = max(float(log_joint[index]) for index in compatible_indices)
    raw = {
        index: float(np.exp(float(log_joint[index]) - maximum))
        for index in compatible_indices
    }
    denominator = float(sum(raw.values()))
    result: dict[int, np.ndarray] = {}
    for position, support in enumerate(supports):
        if position in visible:
            continue
        marginal = np.zeros(support.size, dtype=np.float64)
        for candidate_index, token_id in enumerate(support):
            marginal[candidate_index] = sum(
                raw[index]
                for index in compatible_indices
                if int(sequences[index, position]) == int(token_id)
            )
        marginal /= denominator
        result[position] = marginal
    return result


def verify_enumeration_agreement(
    sequences: np.ndarray,
    log_joint: np.ndarray,
    visible: dict[int, int],
    supports: tuple[np.ndarray, ...],
) -> float:
    """Écart maximal entre les deux implémentations du posterior exact."""
    vectorized = exact_token_posteriors(sequences, log_joint, visible, supports)
    reference = exact_token_posteriors_reference(sequences, log_joint, visible, supports)
    if vectorized.keys() != reference.keys():
        raise ValueError("les deux énumérations ne couvrent pas les mêmes positions")
    maximum = 0.0
    for position in vectorized:
        maximum = max(
            maximum,
            float(np.max(np.abs(vectorized[position] - reference[position]))),
        )
    return maximum


def kl_exact_to_target(exact: np.ndarray, target: np.ndarray) -> float:
    """Calcule ``KL(exact || target)`` avec la convention de l'équation 28."""
    exact_values = np.asarray(exact, dtype=np.float64)
    target_values = np.asarray(target, dtype=np.float64)
    if exact_values.shape != target_values.shape:
        raise ValueError("exact et target doivent avoir la même forme")
    if np.any(exact_values < 0.0) or np.any(target_values < 0.0):
        raise ValueError("les distributions doivent être non négatives")
    positive = exact_values > 0.0
    if np.any(target_values[positive] <= 0.0):
        raise ValueError("la cible est nulle sous une probabilité exacte positive")
    terms = exact_values[positive] * (
        np.log(exact_values[positive]) - np.log(target_values[positive])
    )
    value = float(np.sum(terms))
    if not np.isfinite(value):
        raise ValueError("KL non finie")
    return value


def sample_compatible_index(
    log_weights: np.ndarray,
    compatible: np.ndarray,
    rng: np.random.Generator,
) -> int:
    """Échantillonne un index selon des poids log, restreint aux compatibles."""
    weights = np.asarray(log_weights, dtype=np.float64)
    mask = np.asarray(compatible, dtype=bool)
    if weights.ndim != 1 or mask.shape != weights.shape:
        raise ValueError("log_weights et compatible doivent être des vecteurs alignés")
    if not np.any(mask):
        raise ValueError("aucune séquence compatible")
    shifted = weights[mask] - np.max(weights[mask])
    probabilities = np.exp(shifted)
    probabilities /= np.sum(probabilities)
    local_index = int(rng.choice(probabilities.size, p=probabilities))
    compatible_indices = np.flatnonzero(mask)
    return int(compatible_indices[local_index])


def _logsumexp(values: np.ndarray) -> float:
    maximum = float(np.max(values))
    return maximum + float(np.log(np.sum(np.exp(values - maximum))))


def _normalized_from_logs(log_values: np.ndarray) -> np.ndarray:
    shifted = log_values - np.max(log_values)
    probabilities = np.exp(shifted)
    total = float(np.sum(probabilities))
    if total <= 0.0 or not np.isfinite(total):
        raise ValueError("normalisation impossible")
    return probabilities / total


def all_support_combinations(supports: tuple[np.ndarray, ...]) -> np.ndarray:
    """Produit la liste cartésienne des supports, pour les contrôles de test."""
    return np.asarray(list(product(*(tuple(int(x) for x in support) for support in supports))), dtype=np.int64)


# ---------------------------------------------------------------------------
# Synthèse, bootstrap et décision de porte (protocole J14, DPDD_SPEC §6.9)
# ---------------------------------------------------------------------------


def summarize_marginals(records: list[dict]) -> dict:
    """Moyennes de KL causale, future-aware et de leur différence appariée.

    L'agrégation est la moyenne arithmétique sur les marginaux masqués
    (équation 28 et §6.9.1 de l'Annexe D).
    """
    if not records:
        return {
            "n": 0,
            "mean_causal_kl": None,
            "mean_future_kl": None,
            "mean_delta_kl": None,
        }
    causal = np.asarray([record["causal_kl"] for record in records], dtype=np.float64)
    future = np.asarray([record["future_kl"] for record in records], dtype=np.float64)
    delta = causal - future
    return {
        "n": len(records),
        "mean_causal_kl": float(np.mean(causal)),
        "mean_future_kl": float(np.mean(future)),
        "mean_delta_kl": float(np.mean(delta)),
    }


def bootstrap_delta_ci(
    records: list[dict],
    seed: int,
    iterations: int,
    group_fields: tuple[str, ...] = ("config_index", "block_index"),
) -> tuple[float, float]:
    """IC 95 % de la réduction moyenne de KL, par bootstrap de blocs de base.

    Un bloc de base (un état masqué et ses quatre niveaux de bruit) est l'unité
    de rééchantillonnage : la dépendance entre ses marginaux est préservée.
    Les blocs sont rééchantillonnés indépendamment dans chaque configuration,
    puis les marginaux rééchantillonnés sont pooled comme dans l'estimateur de
    la moyenne arithmétique du papier.
    """
    if not records:
        return (float("nan"), float("nan"))
    if iterations <= 0:
        raise ValueError("iterations doit être strictement positif")
    rng = np.random.default_rng(seed)
    groups: dict[tuple, list[float]] = {}
    for record in records:
        key = tuple(record[field] for field in group_fields)
        groups.setdefault(key, []).append(record["causal_kl"] - record["future_kl"])
    by_config: dict[tuple, list[list[float]]] = {}
    for key, values in groups.items():
        by_config.setdefault(key[:-1], []).append(values)
    samples = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        pooled: list[float] = []
        for config_blocks in by_config.values():
            count = len(config_blocks)
            selected = rng.integers(0, count, size=count)
            for block_index in selected:
                pooled.extend(config_blocks[block_index])
        samples[index] = float(np.mean(pooled))
    return (
        float(np.percentile(samples, 2.5)),
        float(np.percentile(samples, 97.5)),
    )


@dataclass(frozen=True)
class GateDecision:
    """Décision de porte J14 et motifs détaillés.

    Les champs ``reserve_configurations`` et ``reserve_noise_levels`` nomment
    les configurations et les niveaux de bruit dont la réduction moyenne de KL
    est négative : une réserve est obligatoire dans ce cas, même si le
    critère primaire est favorable.
    """

    status: str
    mean_delta_kl: float
    primary_ci95: tuple[float, float]
    total_marginals: int
    blocks_per_configuration: dict[str, int]
    primary_delta_positive: bool
    primary_ci_excludes_zero: bool
    minimum_marginals_met: bool
    numeric_integrity_met: bool
    reserve_configurations: tuple[str, ...]
    reserve_noise_levels: tuple[str, ...]

    @property
    def reserve_required(self) -> bool:
        """Vrai si une configuration ou un niveau de bruit dégrade la KL."""
        return bool(self.reserve_configurations or self.reserve_noise_levels)


def evaluate_gate(
    records: list[dict],
    *,
    seed: int,
    iterations: int,
    minimum_marginals: int = 2000,
    minimum_blocks_per_configuration: int = 100,
    numerical_max_error: float = 0.0,
    max_distribution_error: float = 0.0,
) -> GateDecision:
    """Applique les critères d'acceptation pré-enregistrés (§6.9.3–6.9.4).

    La porte est franchie si et seulement si :

    1. la réduction agrégée est positive et son IC 95 % exclut zéro ;
    2. chaque configuration compte au moins ``minimum_blocks_per_configuration``
       blocs de base distincts et l'ensemble compte au moins
       ``minimum_marginals`` marginaux ;
    3. l'intégrité numérique est respectée (énumération de référence, sommes
       de distribution, valeurs finies).

    Une configuration ou un niveau de bruit dont la réduction moyenne est
    négative impose le statut « franchie avec réserve ».
    """
    summary = summarize_marginals(records)
    primary_ci = bootstrap_delta_ci(records, seed, iterations)
    distinct: dict[str, set] = {}
    for record in records:
        distinct.setdefault(str(record["config_index"]), set()).add(record["block_index"])
    blocks_per_configuration = {name: len(values) for name, values in distinct.items()}

    finite = all(
        np.isfinite(value)
        for record in records
        for value in (record["causal_kl"], record["future_kl"])
    )
    numeric_ok = (
        finite
        and numerical_max_error <= 1e-12
        and max_distribution_error <= 1e-5
    )
    primary_positive = summary["mean_delta_kl"] is not None and summary["mean_delta_kl"] > 0.0
    primary_ci_ok = bool(np.isfinite(primary_ci[0]) and primary_ci[0] > 0.0)
    blocks_ok = bool(blocks_per_configuration) and all(
        count >= minimum_blocks_per_configuration for count in blocks_per_configuration.values()
    )
    marginals_ok = len(records) >= minimum_marginals

    reserve_configurations = tuple(
        sorted(
            name
            for name in sorted(distinct)
            if summarize_marginals(
                [record for record in records if str(record["config_index"]) == name]
            )["mean_delta_kl"]
            < 0.0
        )
    )
    reserve_noise_levels = tuple(
        sorted(
            str(noise)
            for noise in sorted({record["noise"] for record in records})
            if summarize_marginals(
                [record for record in records if record["noise"] == noise]
            )["mean_delta_kl"]
            < 0.0
        )
    )

    if primary_positive and primary_ci_ok and blocks_ok and marginals_ok and numeric_ok:
        status = "passed_with_reserve" if (reserve_configurations or reserve_noise_levels) else "passed"
    else:
        status = "failed"

    return GateDecision(
        status=status,
        mean_delta_kl=float(summary["mean_delta_kl"]),
        primary_ci95=primary_ci,
        total_marginals=len(records),
        blocks_per_configuration=blocks_per_configuration,
        primary_delta_positive=primary_positive,
        primary_ci_excludes_zero=primary_ci_ok,
        minimum_marginals_met=blocks_ok and marginals_ok,
        numeric_integrity_met=numeric_ok,
        reserve_configurations=reserve_configurations,
        reserve_noise_levels=reserve_noise_levels,
    )

# ---------------------------------------------------------------------------
# Synthèse multi-prompt, test non paramétrique et écart cible tronquée (J15)
# ---------------------------------------------------------------------------


def _normal_cdf(value: float) -> float:
    """Fonction de répartition de la loi normale centrée réduite (stdlib seul)."""
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def per_prompt_marginals(records: list[dict]) -> dict[str, dict]:
    """Moyenne et médiane de ``ΔKL`` par prompt.

    La médiane complète la moyenne arithmétique du critère primaire : elle est
    robuste à la queue (règle de lecture R4 de J14) et permet de confirmer que
    l'effet n'est pas porté par une poignée de marginaux favorables.
    """
    by_prompt: dict[str, list[dict]] = {}
    for record in records:
        by_prompt.setdefault(str(record["prompt"]), []).append(record)
    result: dict[str, dict] = {}
    for prompt, group in by_prompt.items():
        deltas = np.asarray([record["delta_kl"] for record in group], dtype=np.float64)
        result[prompt] = {
            "n": len(group),
            "mean_delta_kl": float(np.mean(deltas)),
            "median_delta_kl": float(np.median(deltas)),
        }
    return result


def bootstrap_prompt_ci(
    records: list[dict],
    *,
    seed: int,
    iterations: int,
) -> dict[str, tuple[float, float]]:
    """IC 95 % de la ``ΔKL`` moyenne par prompt, par bootstrap de blocs.

    L'unité de rééchantillonnage est le bloc de base (identifié par
    ``(config_index, block_index)``) dans chaque prompt : la variabilité
    inter-prompt est ainsi mesurée sans mélanger les prompts entre eux. Un IC
    qui exclut zéro pour un prompt indique un effet significatif sur ce seul
    prompt.
    """
    if iterations <= 0:
        raise ValueError("iterations doit être strictement positif")
    blocks_by_prompt: dict[str, dict[tuple, list[float]]] = {}
    for record in records:
        block_key = (str(record["config_index"]), str(record["block_index"]))
        delta = record["causal_kl"] - record["future_kl"]
        blocks_by_prompt.setdefault(str(record["prompt"]), {}).setdefault(block_key, []).append(delta)
    prompts = sorted(blocks_by_prompt)
    rng = np.random.default_rng(seed)
    result: dict[str, tuple[float, float]] = {}
    for prompt in prompts:
        block_lists = list(blocks_by_prompt[prompt].values())
        samples = np.empty(iterations, dtype=np.float64)
        for index in range(iterations):
            count = len(block_lists)
            selected = rng.integers(0, count, size=count)
            pooled: list[float] = []
            for block_index in selected:
                pooled.extend(block_lists[block_index])
            samples[index] = float(np.mean(pooled))
        result[prompt] = (
            float(np.percentile(samples, 2.5)),
            float(np.percentile(samples, 97.5)),
        )
    return result


def sign_test(deltas: np.ndarray) -> dict[str, object]:
    """Test des signes (non paramétrique), approximation normale à deux queues.

    Sous l'hithèse nulle (médiane de ``ΔKL`` nulle), le nombre de signes
    positifs une binomiale ``(n, 0,5)``. L'approximation normale avec
    correction de continuité donne une valeur en p sans hypothèse sur la forme
    de la distribution : complément non paramétrique au critère primaire fondé
    sur la moyenne.
    """
    values = np.asarray(deltas, dtype=np.float64)
    positive = int(np.sum(values > 0.0))
    negative = int(np.sum(values < 0.0))
    effective = positive + negative
    if effective == 0:
        return {"n": 0, "positive": 0, "negative": 0, "p_value": float("nan")}
    expected = effective / 2.0
    std = math.sqrt(effective / 4.0)
    z = (abs(positive - expected) - 0.5) / std
    p_value = 2.0 * (1.0 - _normal_cdf(z))
    return {
        "n": int(effective),
        "positive": positive,
        "negative": negative,
        "p_value": float(max(0.0, min(1.0, p_value))),
    }


def truncate_causal_gap(
    full_causal: np.ndarray,
    k: int,
) -> dict[str, float]:
    """Écart entre la cible causale complète et sa troncade aux ``k`` meilleurs tokens.

    Deux scalaires finis, décisionnels avant toute implémentation d'un
    ``TeacherScorer`` :

    - ``discarded_mass`` : masse de probabilité que la troncature met hors du
      support (1 − somme des ``k`` plus grandes probabilités) ;
    - ``tv_distance`` : distance L1 normalisée (distance de variation totale)
      entre la distribution complète et la version tronquée renormalisée.

    Une troncature à ``k`` grande échelle ne doit ni perdre de masse notable,
    ni éloigner la cible de la distribution complète. La KL par rapport au
    posterior exact n'est pas calculée ici (infinie quand le support exact
    dépasse le top-`k`) : la distance de variation totale l'englobe et reste
    finie.
    """
    probs = np.asarray(full_causal, dtype=np.float64)
    total = float(np.sum(probs))
    if total <= 0.0 or not np.isfinite(total):
        return {"discarded_mass": float("nan"), "tv_distance": float("nan")}
    normalized = probs / total
    k = min(k, normalized.size)
    if k >= normalized.size:
        return {"discarded_mass": 0.0, "tv_distance": 0.0}
    top_indices = np.argpartition(-normalized, k - 1)[:k]
    truncated = np.zeros_like(normalized)
    truncated[top_indices] = normalized[top_indices]
    discarded_mass = float(1.0 - np.sum(truncated))
    tv_distance = float(np.sum(np.abs(normalized - truncated))) / 2.0
    return {"discarded_mass": discarded_mass, "tv_distance": tv_distance}


def evaluate_multiprompt(
    records: list[dict],
    *,
    seed: int,
    iterations: int,
) -> dict:
    """Statistiques du diagnostic multi-prompt (§6.9.7).

    Agrégat (critère primaire sur tous les marginaux, IC par bootstrap de blocs
    distincts), statistiques per-prompt (moyenne/médiane + IC 95 %), et test
    non paramétrique (médiane + test des signes). C'est l'agrégateur qui alimente
    la décision GO / GO conditionnel / NO-GO de J15.
    """
    summary = summarize_marginals(records)
    deltas = np.asarray([record["delta_kl"] for record in records], dtype=np.float64)
    # Blocs distincts across prompts : la clé d'agrégation inclut ``prompt`` pour
    # ne pas confondre les block_index d'après prompts différents.
    aggregate_ci = bootstrap_delta_ci(
        records,
        seed,
        iterations,
        group_fields=("prompt", "config_index", "block_index"),
    )
    per_prompt = per_prompt_marginals(records)
    prompt_ci = bootstrap_prompt_ci(records, seed=seed, iterations=iterations)
    if deltas.size:
        sig = sign_test(deltas)
    else:
        sig = {"n": 0, "positive": 0, "negative": 0, "p_value": float("nan")}
    prompts_excluding_zero = sum(
        1
        for lower, upper in prompt_ci.values()
        if (np.isfinite(lower) and lower > 0.0) or (np.isfinite(upper) and upper < 0.0)
    )
    n_prompts = len(prompt_ci)
    return {
        "n": int(summary["n"]),
        "mean_delta_kl": summary["mean_delta_kl"],
        "median_delta_kl": float(np.median(deltas)) if deltas.size else float("nan"),
        "aggregate_ci95_delta_kl": list(aggregate_ci),
        "per_prompt": per_prompt,
        "prompt_ci95": {prompt: list(ci) for prompt, ci in prompt_ci.items()},
        "sign_test": sig,
        "n_prompts": n_prompts,
        "n_prompts_ci_excludes_zero": prompts_excluding_zero,
        "proportion_prompts_significant": (prompts_excluding_zero / n_prompts) if n_prompts else float("nan"),
    }
