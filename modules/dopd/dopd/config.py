"""Configuration du module d-OPD (arXiv:2609.35362).

Les valeurs par défaut sont celles du papier (section 4 et appendices C) :

* taille de bloc ``N = 4`` pour les comparaisons d'échelle ;
* ``k = 16`` candidats par distribution (union professeur/élève) ;
* porte de correction activée (vérificateur de tâche) ;
* quatre pas de débruitage par bloc, réponse maximale de 4 000 tokens ;
* scorer dans le dtype natif du checkpoint, arithmétique en float32.

Ce fichier ne dépend que de la bibliothèque standard : il est importable sans
torch, sans transformers et sans GPU.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

#: Version du paquet ``academiia.dopd``. Source de vérité unique : elle est
#: reprise dans le bloc ``producer.version`` du manifeste de lignée ADR-002.
PACKAGE_VERSION = "0.1.0"

#: dtype d'arithmétique par défaut (App. B du papier) : scores relatifs,
#: facteurs de correction et normaliseur plein vocabulaire sont calculés en
#: float32. C'est la seule source de vérité du module pour ce réglage :
#: :func:`academiia.dopd.compute_future_aware_target` l'utilise comme défaut.
DEFAULT_ARITHMETIC_DTYPE = "float32"

#: dtypes d'arithmétique acceptés par la brique de cible.
SUPPORTED_ARITHMETIC_DTYPES = ("float32", "float64")


@dataclass(frozen=True)
class DOPDConfig:
    """Hyperparamètres de d-OPD, avec les valeurs par défaut du papier.

    Les champs décrivent la **cible** et le **rollout**, pas l'optimiseur :
    l'optimiseur (AdamW, lr, warmup, ZeRO) appartient au futur harnais
    d'entraînement et sera spécifié en J13+.

    Attributes:
        block_size: nombre de tokens par bloc (``N``), ``4`` par défaut.
        candidate_k: nombre de candidats top-k par distribution ; l'union en
            contient au plus ``2 * candidate_k``.
        correctness_gate: appliquer la correction seulement aux rollouts
            vérifiés (``True``, défaut du papier).
        denoising_steps: pas de débruitage par bloc pendant le rollout.
        max_response_length: longueur maximale d'une réponse générée.
        scorer_dtype: dtype natif du checkpoint du professeur.
        arithmetic_dtype: dtype des calculs de scores et de normalisation.
            **Consommé** par :func:`academiia.dopd.compute_future_aware_target`,
            qui l'utilise comme valeur par défaut.
    """

    block_size: int = 4
    candidate_k: int = 16
    correctness_gate: bool = True
    denoising_steps: int = 4
    max_response_length: int = 4_000
    scorer_dtype: str = "bfloat16"
    arithmetic_dtype: str = DEFAULT_ARITHMETIC_DTYPE

    def __post_init__(self) -> None:
        for name in ("block_size", "candidate_k", "denoising_steps", "max_response_length"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} doit être un entier, reçu {type(value).__name__}")
            if value < 1:
                raise ValueError(f"{name} doit être >= 1, reçu {value}")
        if not isinstance(self.correctness_gate, bool):
            raise TypeError("correctness_gate doit être un booléen")
        for name in ("scorer_dtype", "arithmetic_dtype"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} doit être une chaîne non vide")
        if self.arithmetic_dtype not in SUPPORTED_ARITHMETIC_DTYPES:
            raise ValueError(
                "arithmetic_dtype doit être l'un de "
                f"{SUPPORTED_ARITHMETIC_DTYPES}, reçu {self.arithmetic_dtype!r}"
            )

    @property
    def candidate_union_bound(self) -> int:
        """Borne supérieure de la taille de l'union des top-k (sans le référent)."""
        return 2 * self.candidate_k

    @property
    def max_scorer_branches(self) -> int:
        """Nombre maximal de branches de scoring : union (≤ 2k) + référence."""
        return self.candidate_union_bound + 1

    def to_dict(self) -> dict:
        """Représentation sérialisable, destinée au futur manifeste de lignée."""
        return asdict(self)
