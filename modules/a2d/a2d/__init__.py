"""Bibliotheque A2D conforme (AR-to-Diffusion Transfer / Merge).

Point d'entree : `scripts/merge_a2d.py` (CLI) et `scripts/verify_merge.py`.

Principes (A2D, arXiv:2610.08108) :
  - tau_A = A_p - A_b, tau_D = D_p - D_b (§2.1)
  - A2D-Transfer : D_b + alpha*tau_A (eq. 1, §2.4)
  - A2D-Merge    : D_b + (1-lambda)*tau_D + lambda*tau_A
                 = D_p + lambda*(tau_A - tau_D) (eq. 2, §2.4)
  - arithmetique FP32 sequentielle + UN SEUL cast BF16 (App. A.3)
  - tau calcule a la volee : jamais persiste sur disque.
"""

from .adapters import (ADAPTER_DTYPES, AdapterError, scale_adapter,
                       verify_scaled_adapter)
from .merge import (COVERAGE_POLICIES, MODE_ALIASES, MODES, CoverageError, Expected,
                    MergeConfig, MergeError, MergePolicy, MergeRefused, MergeResult,
                    MissingVectorError, PlanEntry, ShapeMismatchError,
                    build_plan, compute_effective_coefficients, effective_coefficient,
                    expected_tensor, merge_checkpoint, mode_coefficients,
                    normalize_mode, resolve_policy, validate_config,
                    verify_checkpoint, verify_written)
from .naming import (ALL_FAMILIES, FAMILY_ATTN, FAMILY_EMBED, FAMILY_EXPERTS,
                     FAMILY_HEAD, FAMILY_MLP, FAMILY_NORMS, FAMILY_OTHER,
                     FAMILY_ROUTER, GEMMA4_TO_DIFFUSIONGEMMA, IDENTITY, MAPPINGS,
                     NameMapping, family_of, get_mapping, module_of)
from .sources import (CheckpointPairSource, LegacyNpySource, LoraAdapterSource,
                      VectorSource, index_fingerprint, load_weight_map)

__all__ = [
    "ADAPTER_DTYPES", "AdapterError", "scale_adapter", "verify_scaled_adapter",
    "COVERAGE_POLICIES", "MODE_ALIASES", "MODES", "CoverageError", "Expected",
    "MergeConfig", "MergeError", "MergePolicy", "MergeRefused", "MergeResult",
    "MissingVectorError", "PlanEntry", "ShapeMismatchError", "build_plan",
    "compute_effective_coefficients", "effective_coefficient", "expected_tensor",
    "merge_checkpoint", "mode_coefficients", "normalize_mode", "resolve_policy",
    "validate_config", "verify_checkpoint", "verify_written",
    "ALL_FAMILIES", "GEMMA4_TO_DIFFUSIONGEMMA", "IDENTITY", "MAPPINGS", "NameMapping",
    "family_of", "get_mapping", "module_of",
    "FAMILY_ATTN", "FAMILY_EMBED", "FAMILY_EXPERTS", "FAMILY_HEAD", "FAMILY_MLP",
    "FAMILY_NORMS", "FAMILY_OTHER", "FAMILY_ROUTER",
    "CheckpointPairSource", "LegacyNpySource", "LoraAdapterSource", "VectorSource",
    "index_fingerprint", "load_weight_map",
]
