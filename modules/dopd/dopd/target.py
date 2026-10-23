"""Cible future-aware de d-OPD (équations 15-18), implémentation de référence CPU.

Cette brique est volontairement **pure** : elle reçoit une distribution causale
déjà calculée par le professeur, les scores de suffixe déjà calculés pour un
petit ensemble de candidats, et rend la distribution corrigée. Elle ne fait
aucun forward, ne lit aucun checkpoint, n'utilise ni torch ni tokenizer et
n'écrit rien sur disque.

La correction suit le papier (arXiv:2609.35362, §4.1-4.2) :

* seuls les candidats de l'union top-k professeur/élève sont repondérés ;
* les tokens hors candidats gardent un facteur de correction **1** ;
* les scores sont exprimés relativement au token de référence ``x̂_i`` ;
* si le normaliseur n'est pas fini et strictement positif, la cible causale
  est conservée (fallback) ;
* les zéros exacts du prior restent nuls.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

REASON_APPLIED = "applied"
REASON_DISABLED = "disabled"
REASON_NO_VISIBLE_FUTURE = "no_visible_future"
REASON_EMPTY_CANDIDATES = "empty_candidate_set"
REASON_FALLBACK = "non_finite_normalizer"

__all__ = [
    "REASON_APPLIED",
    "REASON_DISABLED",
    "REASON_EMPTY_CANDIDATES",
    "REASON_FALLBACK",
    "REASON_NO_VISIBLE_FUTURE",
    "TargetResult",
    "candidate_union",
    "compute_future_aware_target",
    "relative_future_deltas",
]


@dataclass(frozen=True)
class TargetResult:
    """Résultat d'une construction de cible future-aware.

    Attributes:
        target: distribution corrigée (ou copie du prior causal en cas de
            fallback), de même longueur que le prior et de somme 1.
        applied: ``True`` si la correction future-aware a été appliquée.
        reason: motif lisible ; l'une des constantes ``REASON_*``.
        n_candidates: nombre de candidats scorés (0 si aucune correction).
    """

    target: np.ndarray
    applied: bool
    reason: str
    n_candidates: int


def _as_prior(causal_prior: Sequence[float] | np.ndarray) -> np.ndarray:
    """Valide et copie une distribution causale (float64, somme ≈ 1)."""
    prior = np.asarray(causal_prior, dtype=np.float64)
    if prior.ndim != 1:
        raise ValueError(f"causal_prior doit être 1-D, reçu {prior.ndim}-D")
    if prior.size == 0:
        raise ValueError("causal_prior est vide")
    if not np.all(np.isfinite(prior)):
        raise ValueError("causal_prior contient NaN ou ±inf")
    if np.any(prior < 0.0):
        raise ValueError("causal_prior contient une probabilité négative")
    total = float(prior.sum())
    if total <= 0.0:
        raise ValueError("causal_prior est de masse nulle")
    # Tolérance volontairement large : un softmax float32 sur un grand
    # vocabulaire peut dériver de quelques 1e-5. On renormalise ensuite.
    if not np.isclose(total, 1.0, rtol=0.0, atol=1e-4):
        raise ValueError(
            f"causal_prior doit sommer à 1 (tolérance 1e-4), somme reçue {total:.8f}"
        )
    # Si la somme est déjà 1 à la précision machine, on conserve les valeurs
    # exactes : le fallback « p̃ = r_i » doit être bit-à-bit le prior.
    if np.isclose(total, 1.0, rtol=0.0, atol=1e-12):
        return prior.copy()
    return prior / total


def _as_candidates(
    candidate_indices: Sequence[int] | np.ndarray, vocab_size: int
) -> np.ndarray:
    """Valide les indices de candidats : 1-D, entiers, uniques, dans le vocabulaire."""
    candidates = np.asarray(candidate_indices)
    if candidates.ndim != 1:
        raise ValueError(f"candidate_indices doit être 1-D, reçu {candidates.ndim}-D")
    if candidates.size == 0:
        return np.empty(0, dtype=np.int64)
    if not np.issubdtype(candidates.dtype, np.integer):
        raise TypeError("candidate_indices doit contenir des entiers")
    candidates = candidates.astype(np.int64, copy=False)
    if int(candidates.min()) < 0 or int(candidates.max()) >= vocab_size:
        raise IndexError(
            f"candidate_indices hors vocabulaire [0, {vocab_size}[: "
            f"min={int(candidates.min())}, max={int(candidates.max())}"
        )
    if np.unique(candidates).size != candidates.size:
        raise ValueError("candidate_indices contient des doublons")
    return candidates


def _as_scores(scores: Sequence[float] | np.ndarray, expected_size: int) -> np.ndarray:
    """Valide des log-vraisemblances : taille attendue, pas de NaN, pas de +inf."""
    values = np.asarray(scores, dtype=np.float64)
    if values.shape != (expected_size,):
        raise ValueError(
            f"shape des scores {values.shape} incompatible avec {expected_size} candidats"
        )
    if np.any(np.isnan(values)):
        raise ValueError("les scores contiennent NaN")
    if np.any(np.isposinf(values)):
        raise ValueError("les scores contiennent +inf")
    return values


def relative_future_deltas(
    loglikelihoods: Sequence[float] | np.ndarray,
    reference_loglikelihood: float,
) -> np.ndarray:
    """Calcule ``Δs_i(v) = s_i(v) − s_i(x̂_i)`` (eq. 16).

    Le score de référence doit être fini : une valeur ``-inf`` rendrait les
    deltas non finis et la normalisation impossible. Un score candidat peut
    valoir ``-inf`` (suffixe de probabilité nulle) : son delta vaut alors
    ``-inf`` et son poids corrigé est zéro, ce qui est le comportement voulu.
    """
    if reference_loglikelihood is None:
        raise ValueError("reference_loglikelihood ne peut pas être None")
    reference = float(reference_loglikelihood)
    if not np.isfinite(reference):
        raise ValueError("reference_loglikelihood doit être fini")
    values = np.asarray(loglikelihoods, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"loglikelihoods doit être 1-D, reçu {values.ndim}-D")
    values = _as_scores(values, values.size)
    deltas = values - reference
    if np.any(np.isnan(deltas)) or np.any(np.isposinf(deltas)):
        raise ValueError("les deltas contiennent NaN ou +inf")
    return deltas


def candidate_union(
    teacher_topk: Sequence[int] | np.ndarray,
    student_topk: Sequence[int] | np.ndarray,
) -> np.ndarray:
    """Union triée et sans doublon des top-k professeur et élève (§4.2).

    L'ordre des indices n'a pas d'importance pour la cible ; le tri rend les
    tests et les journaux déterministes.
    """
    teacher = np.asarray(teacher_topk)
    student = np.asarray(student_topk)
    for name, values in (("teacher_topk", teacher), ("student_topk", student)):
        if values.ndim != 1:
            raise ValueError(f"{name} doit être 1-D")
        if values.size and not np.issubdtype(values.dtype, np.integer):
            raise TypeError(f"{name} doit contenir des entiers")
    if teacher.size == 0 and student.size == 0:
        return np.empty(0, dtype=np.int64)
    merged = np.concatenate(
        [teacher.astype(np.int64, copy=False), student.astype(np.int64, copy=False)]
    )
    return np.unique(merged)


def compute_future_aware_target(
    causal_prior: Sequence[float] | np.ndarray,
    *,
    candidate_indices: Sequence[int] | np.ndarray | None = None,
    candidate_loglikelihoods: Sequence[float] | np.ndarray | None = None,
    reference_loglikelihood: float | None = None,
    visible_future: bool = True,
    apply: bool = True,
) -> TargetResult:
    """Construit la cible ``p̃_T,i`` de l'équation 18.

    Args:
        causal_prior: ``r_i(v)``, distribution causale du professeur sur tout le
            vocabulaire (1-D, probabilités, somme 1 à 1e-4 près).
        candidate_indices: indices de ``V_i`` (union top-k, sans doublon).
        candidate_loglikelihoods: ``s_i(v)`` pour chaque candidat de ``V_i``.
        reference_loglikelihood: ``s_i(x̂_i)``, score du token produit par le
            rollout. Il peut ne pas appartenir à ``V_i``.
        visible_future: ``False`` si aucun token visible à droite de ``i`` ;
            la cible se réduit alors au prior causal (eq. 12, cas limite).
        apply: ``False`` pour désactiver la correction (porte de correction
            fermée, par exemple rollout incorrect). Le prior est rendu tel quel.

    Returns:
        Un :class:`TargetResult`. En cas de fallback, ``target`` est une copie
        du prior causal renormalisé et ``applied`` vaut ``False``.

    Raises:
        ValueError: entrée invalide (prior, candidats, scores, référence).
        TypeError: indices non entiers.
        IndexError: indice de candidat hors vocabulaire.
    """
    prior = _as_prior(causal_prior)

    if not apply:
        return TargetResult(prior, False, REASON_DISABLED, 0)
    if not visible_future:
        return TargetResult(prior, False, REASON_NO_VISIBLE_FUTURE, 0)

    if candidate_indices is None:
        raise ValueError("candidate_indices est requis quand la correction est active")
    candidates = _as_candidates(candidate_indices, prior.size)
    if candidates.size == 0:
        return TargetResult(prior, False, REASON_EMPTY_CANDIDATES, 0)
    if candidate_loglikelihoods is None:
        raise ValueError("candidate_loglikelihoods est requis quand la correction est active")
    if reference_loglikelihood is None:
        raise ValueError("reference_loglikelihood est requis quand la correction est active")

    scores = _as_scores(candidate_loglikelihoods, candidates.size)
    deltas = relative_future_deltas(scores, reference_loglikelihood)

    # Calcul en log-espace pour éviter les débordements de exp(Δs).
    log_weights = np.full(prior.size, -np.inf, dtype=np.float64)
    positive = prior > 0.0
    log_weights[positive] = np.log(prior[positive])
    log_weights[candidates] += deltas

    max_log_weight = float(np.max(log_weights))
    if not np.isfinite(max_log_weight):
        return TargetResult(prior, False, REASON_FALLBACK, int(candidates.size))

    weights = np.exp(log_weights - max_log_weight)
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0.0:
        return TargetResult(prior, False, REASON_FALLBACK, int(candidates.size))

    target = weights / total
    if not np.all(np.isfinite(target)):
        return TargetResult(prior, False, REASON_FALLBACK, int(candidates.size))

    return TargetResult(target, True, REASON_APPLIED, int(candidates.size))
