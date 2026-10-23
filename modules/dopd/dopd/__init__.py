"""d-OPD — Future-Aware On-Policy Distillation (arXiv:2609.35362).

Squelette J12 du module. Le paquet contient :

* :class:`DOPDConfig` — hyperparamètres du papier (N = 4, k = 16, porte active) ;
* :func:`compute_future_aware_target` — cible future-aware ``p̃_T,i`` (eq. 15-18),
  en NumPy pur, sans torch ni GPU ;
* :func:`candidate_union` — union des top-k professeur/élève.

Le module n'importe que :mod:`academiia.core` (règle ADR-001) et aucune
bibliothèque lourde. Aucun entraînement, aucun rollout, aucun checkpoint n'est
nécessaire pour l'utiliser ou le tester.

Spécification complète : ``modules/dopd/docs/DOPD_SPEC.md``.
"""

from .config import DOPDConfig
from .target import (
    REASON_APPLIED,
    REASON_DISABLED,
    REASON_EMPTY_CANDIDATES,
    REASON_FALLBACK,
    REASON_NO_VISIBLE_FUTURE,
    TargetResult,
    candidate_union,
    compute_future_aware_target,
    relative_future_deltas,
)

__version__ = "0.1.0"

__all__ = [
    "DOPDConfig",
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
