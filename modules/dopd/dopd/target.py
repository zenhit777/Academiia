"""Cible future-aware de d-OPD (équations 15-18), implémentation de référence CPU.

Cette brique est volontairement **pure** : elle reçoit une distribution causale
déjà calculée par le professeur, les scores de suffixe déjà calculés pour un
petit ensemble de candidats, et rend la distribution corrigée. Elle ne fait
aucun forward, ne lit aucun checkpoint, n'utilise ni torch ni tokenizer et
n'écrit rien sur disque.

La correction suit le papier (arXiv:2609.35362, §4.1-4.2 et App. B) :

* seuls les candidats de l'union top-k professeur/élève sont repondérés ;
* les tokens hors candidats gardent un facteur de correction **1** ;
* les scores sont exprimés relativement au token de référence ``x̂_i`` ;
* les différences de scores, les facteurs de correction, les poids et le
  normaliseur **plein vocabulaire** sont calculés dans le dtype d'arithmétique
  (``float32`` par défaut, App. B) ;
* la correction n'est validée que si le normaliseur plein vocabulaire est fini
  et strictement positif ; sinon la cible causale est conservée (fallback) ;
* les deltas finis ne sont **pas** écrêtés : un débordement ou un
  sous-débitement rend le normaliseur non fini ou nul, donc le fallback ;
* les zéros exacts du prior restent nuls.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .config import DEFAULT_ARITHMETIC_DTYPE, SUPPORTED_ARITHMETIC_DTYPES

REASON_APPLIED = "applied"
REASON_DISABLED = "disabled"
REASON_NO_VISIBLE_FUTURE = "no_visible_future"
REASON_EMPTY_CANDIDATES = "empty_candidate_set"
REASON_NON_FINITE_SCORES = "non_finite_scores"
REASON_FALLBACK = "non_finite_normalizer"

__all__ = [
    "REASON_APPLIED",
    "REASON_DISABLED",
    "REASON_EMPTY_CANDIDATES",
    "REASON_FALLBACK",
    "REASON_NON_FINITE_SCORES",
    "REASON_NO_VISIBLE_FUTURE",
    "TargetResult",
    "candidate_union",
    "compute_future_aware_target",
    "relative_future_deltas",
]

_NUMPY_DTYPES = {"float32": np.float32, "float64": np.float64}


@dataclass(frozen=True)
class TargetResult:
    """Résultat d'une construction de cible future-aware.

    Attributes:
        target: distribution corrigée (ou copie renormalisée du prior causal en
            cas de fallback), de même longueur que le prior et de somme 1 à la
            précision du dtype d'arithmétique près.
        applied: ``True`` si la correction future-aware a été appliquée.
        reason: motif lisible ; l'une des constantes ``REASON_*``.
        n_candidates: nombre de candidats scorés (0 si aucune correction).
    """

    target: np.ndarray
    applied: bool
    reason: str
    n_candidates: int


def _numpy_dtype(arithmetic_dtype: str) -> np.dtype:
    """Traduit un nom de dtype d'arithmétique en dtype NumPy."""
    if arithmetic_dtype not in SUPPORTED_ARITHMETIC_DTYPES:
        raise ValueError(
            "arithmetic_dtype doit être l'un de "
            f"{SUPPORTED_ARITHMETIC_DTYPES}, reçu {arithmetic_dtype!r}"
        )
    return np.dtype(_NUMPY_DTYPES[arithmetic_dtype])


def _as_prior(causal_prior: Sequence[float] | np.ndarray) -> np.ndarray:
    """Valide et normalise une distribution causale (float64, somme 1)."""
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
    # La cible doit sommer à 1 (invariant de TargetResult). On ne dispense de
    # renormalisation que si la somme est *exactement* 1.0 : une somme
    # « presque 1 » (ex. 1 + 5e-13) est renormalisée, sinon le fallback
    # « p̃ = r_i » ne serait pas une distribution.
    if total == 1.0:
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
            f"candidate_indices hors vocabulaire [0, {vocab_size}) : "
            f"min={int(candidates.min())}, max={int(candidates.max())}"
        )
    if np.unique(candidates).size != candidates.size:
        raise ValueError("candidate_indices contient des doublons")
    return candidates


def _as_scores(scores: Sequence[float] | np.ndarray, expected_size: int) -> np.ndarray:
    """Valide la taille des log-vraisemblances.

    Les valeurs non finies ne sont **pas** rejetées ici : elles rendent le
    normaliseur plein vocabulaire non fini, donc le fallback causal (App. B).
    """
    values = np.asarray(scores, dtype=np.float64)
    if values.shape != (expected_size,):
        raise ValueError(
            f"shape des scores {values.shape} incompatible avec {expected_size} candidats"
        )
    return values


def relative_future_deltas(
    loglikelihoods: Sequence[float] | np.ndarray,
    reference_loglikelihood: float,
    *,
    arithmetic_dtype: str = DEFAULT_ARITHMETIC_DTYPE,
) -> np.ndarray:
    """Calcule ``Δs_i(v) = s_i(v) − s_i(x̂_i)`` (eq. 16).

    Le calcul est effectué dans le dtype d'arithmétique (``float32`` par
    défaut, App. B : *relative score differences* sont calculés en float32).

    Les valeurs non finies ne sont pas une erreur de contrat : elles se
    propagent telles quelles dans les deltas. C'est la construction de la
    cible qui décide du **fallback causal** quand elles rendent le normaliseur
    plein vocabulaire non fini. En particulier :

    * un score candidat ``-inf`` (suffixe de probabilité nulle) donne un delta
      ``-inf``, donc un poids corrigé nul — c'est le comportement voulu ;
    * un score candidat ``NaN`` ou ``+inf``, ou une référence non finie, donne
      un delta non fini qui rend le normaliseur invalide.

    Raises:
        ValueError: dimension inattendue, référence ``None`` ou dtype inconnu.
    """
    if reference_loglikelihood is None:
        raise ValueError("reference_loglikelihood ne peut pas être None")
    dtype = _numpy_dtype(arithmetic_dtype)
    values = np.asarray(loglikelihoods, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"loglikelihoods doit être 1-D, reçu {values.ndim}-D")
    values = _as_scores(values, values.size)
    reference = np.asarray(float(reference_loglikelihood), dtype=dtype)
    return values.astype(dtype) - reference


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
    arithmetic_dtype: str = DEFAULT_ARITHMETIC_DTYPE,
) -> TargetResult:
    """Construit la cible ``p̃_T,i`` de l'équation 18.

    La normalisation est celle du papier (App. B) : les poids
    ``r_i(v)·exp(Δs_i(v))`` sont calculés sur **tout le vocabulaire** dans le
    dtype d'arithmétique, et la correction n'est validée que si ce
    normaliseur est fini et strictement positif. Aucun delta fini n'est
    écrêté : un débordement (``exp`` infini) ou un sous-débitement (tous les
    poids nuls) déclenche donc le fallback causal.

    Args:
        causal_prior: ``r_i(v)``, distribution causale du professeur sur tout le
            vocabulaire (1-D, probabilités, somme 1 à 1e-4 près). Elle est
            renormalisée pour que la cible somme 1.
        candidate_indices: indices de ``V_i`` (union top-k, sans doublon).
        candidate_loglikelihoods: ``s_i(v)`` pour chaque candidat de ``V_i``.
        reference_loglikelihood: ``s_i(x̂_i)``, score du token produit par le
            rollout. Il peut ne pas appartenir à ``V_i``. Une valeur non finie
            impose le fallback causal.
        visible_future: ``False`` si aucun token visible à droite de ``i`` ;
            la cible se réduit alors au prior causal (eq. 12, cas limite).
        apply: ``False`` pour désactiver la correction (porte de correction
            fermée, par exemple rollout incorrect). Le prior est rendu tel quel.
        arithmetic_dtype: ``"float32"`` (défaut du papier, App. B) ou
            ``"float64"``.

    Returns:
        Un :class:`TargetResult`. En cas de fallback, ``target`` est une copie
        renormalisée du prior causal et ``applied`` vaut ``False``.

    Raises:
        ValueError: entrée invalide (prior, candidats, scores, référence,
            dtype d'arithmétique).
        TypeError: indices non entiers.
        IndexError: indice de candidat hors vocabulaire.
    """
    dtype = _numpy_dtype(arithmetic_dtype)
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

    # Une référence non finie rend tous les deltas non finis : le ratio
    # exp(Δs) n'a pas de sens, la cible causale est conservée.
    if not np.isfinite(np.asarray(float(reference_loglikelihood), dtype=np.float64)):
        return TargetResult(prior, False, REASON_NON_FINITE_SCORES, int(candidates.size))

    deltas = relative_future_deltas(
        scores, reference_loglikelihood, arithmetic_dtype=arithmetic_dtype
    )
    # Un delta NaN ou +inf rend le normaliseur non fini (App. B). Un delta
    # -inf est légitime : il annule le poids du candidat (suffixe impossible).
    if np.any(np.isnan(deltas)) or np.any(np.isposinf(deltas)):
        return TargetResult(prior, False, REASON_NON_FINITE_SCORES, int(candidates.size))

    # Facteurs de correction : 1 pour les tokens hors candidats, exp(Δs) pour
    # les candidats. Le débordement et le sous-débitement ne sont pas écrêtés :
    # ils sont détectés par le test du normaliseur ci-dessous.
    factors = np.ones(prior.size, dtype=dtype)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        factors[candidates] = np.exp(deltas)

    # Poids plein vocabulaire r_i(v)·facteur(v), en dtype d'arithmétique. Le
    # produit est calculé tel quel : un zéro exact reste nul tant que le
    # facteur est fini, et 0 × ∞ reste non fini — donc déclenche le fallback,
    # comme le prescrit la règle du normaliseur plein vocabulaire.
    prior32 = prior.astype(dtype)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        weights = prior32 * factors

    normalizer = weights.sum(dtype=dtype)
    if not np.isfinite(normalizer) or normalizer <= 0.0:
        return TargetResult(prior, False, REASON_FALLBACK, int(candidates.size))

    target = (weights / normalizer).astype(np.float64)
    if not np.all(np.isfinite(target)):
        return TargetResult(prior, False, REASON_FALLBACK, int(candidates.size))

    return TargetResult(target, True, REASON_APPLIED, int(candidates.size))
