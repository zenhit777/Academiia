"""Code générique partagé — aucune dépendance vers un module.

* :mod:`academiia.core.stats` — IC bootstrap apparié, intervalle de Wilson ;
* :mod:`academiia.core.common` — empreintes SHA-256, index HuggingFace,
  fichiers annexes d'un checkpoint, commit git.
"""

from .common import (git_commit, index_fingerprint, list_source_sidecar_files,
                     sha256_file)
from .stats import bootstrap_ci95, wilson
from .text import degenerate

__all__ = [
    "bootstrap_ci95",
    "degenerate",
    "git_commit",
    "index_fingerprint",
    "list_source_sidecar_files",
    "sha256_file",
    "wilson",
]
