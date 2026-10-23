"""d-OPD — Future-Aware On-Policy Distillation (arXiv:2609.35362).

Paquet ``academiia.dopd``. Il contient :

* :class:`DOPDConfig` — hyperparamètres du papier (N = 4, k = 16, porte active) ;
* :func:`compute_future_aware_target` — cible future-aware ``p̃_T,i`` (eq. 15-18),
  en NumPy pur, sans torch ni GPU ;
* :func:`candidate_union` — union des top-k professeur/élève ;
* le **noyau du diagnostic Annexe D** (:func:`exact_token_posteriors`,
  :func:`evaluate_gate`, ...) — énumération exacte, KL et décision de porte ;
* les **contrats** :class:`TeacherScorer`, :class:`RolloutRecorder`,
  :class:`TaskVerifier`, :class:`LineageWriter` (J13) — abstractions mockables,
  sans scoring ni rollout réels.

Le module n'importe que :mod:`academiia.core` (règle ADR-001) et aucune
bibliothèque lourde. Aucun entraînement, aucun rollout, aucun checkpoint n'est
nécessaire pour l'utiliser ou le tester.

Spécification complète : ``modules/dopd/docs/DOPD_SPEC.md``.
"""

from .config import (
    DEFAULT_ARITHMETIC_DTYPE,
    PACKAGE_VERSION,
    SUPPORTED_ARITHMETIC_DTYPES,
    DOPDConfig,
)
from .diagnostic import (
    GateDecision,
    RestrictedSupport,
    bootstrap_delta_ci,
    evaluate_gate,
    exact_token_posteriors,
    exact_token_posteriors_reference,
    kl_exact_to_target,
    sample_compatible_index,
    summarize_marginals,
    verify_enumeration_agreement,
)
from .interfaces import (
    ARTIFACT_KINDS,
    DEVELOPMENT_MODEL_ID,
    LINEAGE_SCHEMA,
    PRODUCER_MODULE,
    ArtifactRef,
    CompletedBlock,
    DenoisingState,
    LineageManifest,
    LineageWriter,
    RolloutRecorder,
    ScoringRequest,
    ScoringResult,
    TaskVerifier,
    TeacherScorer,
)
from .target import (
    REASON_APPLIED,
    REASON_DISABLED,
    REASON_EMPTY_CANDIDATES,
    REASON_FALLBACK,
    REASON_NON_FINITE_SCORES,
    REASON_NO_VISIBLE_FUTURE,
    TargetResult,
    candidate_union,
    compute_future_aware_target,
    relative_future_deltas,
)

__version__ = PACKAGE_VERSION

__all__ = [
    "ARTIFACT_KINDS",
    "DEFAULT_ARITHMETIC_DTYPE",
    "DEVELOPMENT_MODEL_ID",
    "DOPDConfig",
    "GateDecision",
    "LINEAGE_SCHEMA",
    "PACKAGE_VERSION",
    "PRODUCER_MODULE",
    "REASON_APPLIED",
    "REASON_DISABLED",
    "REASON_EMPTY_CANDIDATES",
    "REASON_FALLBACK",
    "REASON_NON_FINITE_SCORES",
    "REASON_NO_VISIBLE_FUTURE",
    "SUPPORTED_ARITHMETIC_DTYPES",
    "ArtifactRef",
    "CompletedBlock",
    "DenoisingState",
    "LineageManifest",
    "LineageWriter",
    "RestrictedSupport",
    "RolloutRecorder",
    "ScoringRequest",
    "ScoringResult",
    "TargetResult",
    "TaskVerifier",
    "TeacherScorer",
    "bootstrap_delta_ci",
    "candidate_union",
    "compute_future_aware_target",
    "evaluate_gate",
    "exact_token_posteriors",
    "exact_token_posteriors_reference",
    "kl_exact_to_target",
    "relative_future_deltas",
    "sample_compatible_index",
    "summarize_marginals",
    "verify_enumeration_agreement",
]
