"""Contrats d'interface de d-OPD (J13) — stubs mockables, sans torch.

Ce module formalise les interfaces que le papier (arXiv:2609.35362) et le brief
(`modules/dopd/docs/DOPD_SPEC.md` §6.2) laissaient « spécifiées, non
implémentées » :

* :class:`TeacherScorer` — le professeur AR gelé fournit le prior causal
  ``r_i(v)`` (eq. 11) et le score de suffixe ``s_i(v)`` (eq. 14) ;
* :class:`RolloutRecorder` — le rollout de l'élève enregistre les états de
  débruitage visités ``X_t`` (eq. 2) et le bloc complété ``x̂`` (eq. 4) ;
* :class:`TaskVerifier` — l'étiquette de correction ``R(x̂)`` qui ouvre la
  porte de la correction (§3.4) ;
* :class:`LineageWriter` — le manifeste de lignée ADR-002 avec
  ``producer.module = "dopd"``.

Ces classes sont **abstraites** : J13 ne fournit que les contrats, les types de
données et les helpers déterministes (composition du scoring, échantillonnage
d'un état de débruitage, ouverture de la porte, construction du manifeste).
Aucun scoring réel, aucun rollout réel, aucun entraînement : les implémentations
concrètes (Hugging Face, Qwen3-0.6B) vivent dans un harnais **externe** au
paquet pur — règle ADR-001 : ni ``torch``, ni ``transformers`` dans
``academiia.dopd``.

Références :
  - arXiv:2609.35362, §3.1-3.5 et App. A (algorithme), App. B (implémentation) ;
  - `modules/dopd/docs/DOPD_SPEC.md` §2.1 (notation), §3.2 (r_i, s_i),
    §3.4 (porte), §3.5 (algorithme), §6.2 (interfaces), §6.5 (ADR-002) ;
  - `docs/adr/ADR-002-provenance.md` (schéma du manifeste de lignée).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Integral

import numpy as np

from academiia.core import git_commit

from .config import DOPDConfig, PACKAGE_VERSION

#: Identifiant du schéma du manifeste de lignée (ADR-002).
LINEAGE_SCHEMA = "academiia.lineage/1"

#: Module producteur déclaré par d-OPD (ADR-002 : `producer.module`).
PRODUCER_MODULE = "dopd"

#: Types d'artefact acceptés par le manifeste ADR-002.
ARTIFACT_KINDS = ("checkpoint", "adapter", "eval", "report")

#: Checkpoint de développement retenu (DOPD_SPEC §6.3) : plus petite échelle du
#: papier. Ce n'est pas un hyperparamètre : c'est l'identité du checkpoint que
#: le professeur gelé et l'élève initialisent.
DEVELOPMENT_MODEL_ID = "Qwen/Qwen3-0.6B"

__all__ = [
    "ARTIFACT_KINDS",
    "DEVELOPMENT_MODEL_ID",
    "LINEAGE_SCHEMA",
    "PRODUCER_MODULE",
    "ArtifactRef",
    "CompletedBlock",
    "DenoisingState",
    "LineageManifest",
    "LineageWriter",
    "RolloutRecorder",
    "ScoringRequest",
    "ScoringResult",
    "TaskVerifier",
    "TeacherScorer",
]


# ---------------------------------------------------------------------------
# Types de données partagés
# ---------------------------------------------------------------------------


def _as_int_tuple(values: Sequence[int], name: str) -> tuple[int, ...]:
    """Normalise une séquence d'identifiants en tuple d'entiers Python."""
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} doit être une séquence d'entiers, reçu {type(values).__name__}")
    out = tuple(values)
    for value in out:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(
                f"{name} doit contenir des entiers, reçu {type(value).__name__}"
            )
    return tuple(int(value) for value in out)


@dataclass(frozen=True)
class ScoringRequest:
    """Requête de scoring adressée au professeur AR gelé.

    Regroupe les trois quantités dont dépend la cible future-aware à une
    position masquée ``i`` (DOPD_SPEC §3.2-§3.3) :

    * le préfixe causal partagé ``(q, z, Â_i)`` ;
    * le suffixe complété ``Ĉ_i = (x̂_{i+1}, …, x̂_N)`` ;
    * l'ensemble de candidats ``V_i`` et le token de référence ``x̂_i``.

    Attributes:
        prefix_ids: identifiants du préfixe causal partagé ``(q, z, Â_i)``.
        suffix_ids: identifiants du suffixe complété ``Ĉ_i``.
        candidate_ids: tokens candidats ``V_i`` (union des top-k professeur et
            élève, §4.2). Sans doublon, non vide.
        reference_id: token réellement produit par le rollout à la position ``i``,
        ``x̂_i`` ; sert de référence de normalisation (eq. 16-17).
    """

    prefix_ids: tuple[int, ...]
    suffix_ids: tuple[int, ...]
    candidate_ids: tuple[int, ...]
    reference_id: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "prefix_ids", _as_int_tuple(self.prefix_ids, "prefix_ids"))
        object.__setattr__(self, "suffix_ids", _as_int_tuple(self.suffix_ids, "suffix_ids"))
        object.__setattr__(self, "candidate_ids", _as_int_tuple(self.candidate_ids, "candidate_ids"))
        if isinstance(self.reference_id, bool) or not isinstance(self.reference_id, Integral):
            raise TypeError(
                "reference_id doit être un entier, "
                f"reçu {type(self.reference_id).__name__}"
            )
        object.__setattr__(self, "reference_id", int(self.reference_id))
        if not self.candidate_ids:
            raise ValueError("candidate_ids est vide")
        if len(set(self.candidate_ids)) != len(self.candidate_ids):
            raise ValueError("candidate_ids contient des doublons")


@dataclass(frozen=True, eq=False)
class ScoringResult:
    """Résultat d'une requête de scoring au professeur.

    Attributes:
        prior: ``r_i(v) = p_T(v | q, z, Â_i)`` (eq. 11) sur **tout** le
            vocabulaire, de somme 1.
        candidate_ids: tokens scorés, alignés sur ``candidate_loglikelihoods``.
        candidate_loglikelihoods: ``s_i(v) = log p_T(Ĉ_i | q, z, Â_i, v)``
            (eq. 14), aligné sur ``candidate_ids``.
        reference_loglikelihood: ``s_i(x̂_i)``, la référence de normalisation
            (eq. 16).

    Les scores ne sont **pas** filtrés sur leur finitude : une valeur non finie
    n'est pas une erreur du scorer mais un état que la cible sait dégrader en
    cible causale (``REASON_NON_FINITE_SCORES``, DOPD_SPEC §3.6). Seul le prior
    doit être une vraie distribution : il est la base de la renormalisation.
    """

    prior: np.ndarray
    candidate_ids: tuple[int, ...]
    candidate_loglikelihoods: np.ndarray
    reference_loglikelihood: float

    def __post_init__(self) -> None:
        prior = np.asarray(self.prior, dtype=np.float64)
        if prior.ndim != 1:
            raise ValueError(f"prior doit être 1-D, reçu {prior.ndim}-D")
        if prior.size == 0:
            raise ValueError("prior est vide")
        if not np.all(np.isfinite(prior)):
            raise ValueError("prior contient NaN ou ±inf")
        if np.any(prior < 0.0):
            raise ValueError("prior contient une probabilité négative")
        total = float(prior.sum())
        if not np.isclose(total, 1.0, rtol=0.0, atol=1e-4):
            raise ValueError(f"prior doit sommer à 1 (tolérance 1e-4), somme reçue {total:.8f}")
        object.__setattr__(self, "prior", prior)

        candidate_ids = _as_int_tuple(self.candidate_ids, "candidate_ids")
        if not candidate_ids:
            raise ValueError("candidate_ids est vide")
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("candidate_ids contient des doublons")
        object.__setattr__(self, "candidate_ids", candidate_ids)

        scores = np.asarray(self.candidate_loglikelihoods, dtype=np.float64)
        if scores.ndim != 1:
            raise ValueError(
                f"candidate_loglikelihoods doit être 1-D, reçu {scores.ndim}-D"
            )
        if scores.size != len(candidate_ids):
            raise ValueError(
                "candidate_loglikelihoods et candidate_ids doivent avoir la "
                f"même taille ({scores.size} != {len(candidate_ids)})"
            )
        object.__setattr__(self, "candidate_loglikelihoods", scores)

        reference = float(self.reference_loglikelihood)
        object.__setattr__(self, "reference_loglikelihood", reference)


@dataclass(frozen=True)
class DenoisingState:
    """État de débruitage visité ``X_t`` dans un bloc (eq. 2).

    Attributes:
        block_index: index du bloc dans la trajectoire.
        step: pas de débruitage ``t`` (0 = le plus bruité).
        token_ids: jetons du bloc à cet état ; les positions masquées portent
            l'identifiant de masque réservé par le rollout.
        masked_positions: positions ``i`` masquées à cet état ; c'est
            l'ensemble sur lequel la cible est calculée.
    """

    block_index: int
    step: int
    token_ids: tuple[int, ...]
    masked_positions: tuple[int, ...]

    def __post_init__(self) -> None:
        for name in ("block_index", "step"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} doit être un entier, reçu {type(value).__name__}")
            object.__setattr__(self, name, int(value))
        if self.step < 0:
            raise ValueError(f"step doit être >= 0, reçu {self.step}")
        object.__setattr__(self, "token_ids", _as_int_tuple(self.token_ids, "token_ids"))
        masked = _as_int_tuple(self.masked_positions, "masked_positions")
        if len(set(masked)) != len(masked):
            raise ValueError("masked_positions contient des doublons")
        for position in masked:
            if not 0 <= position < len(self.token_ids):
                raise ValueError(
                    f"masked_positions contient {position}, hors de "
                    f"[0, {len(self.token_ids)})"
                )
        object.__setattr__(self, "masked_positions", masked)


@dataclass(frozen=True)
class CompletedBlock:
    """Bloc complété ``x̂`` produit par le rollout (eq. 4).

    Attributes:
        block_index: index du bloc dans la trajectoire.
        token_ids: jetons du bloc une fois complété.
    """

    block_index: int
    token_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if isinstance(self.block_index, bool) or not isinstance(self.block_index, Integral):
            raise TypeError(
                f"block_index doit être un entier, reçu {type(self.block_index).__name__}"
            )
        object.__setattr__(self, "block_index", int(self.block_index))
        object.__setattr__(self, "token_ids", _as_int_tuple(self.token_ids, "token_ids"))


@dataclass(frozen=True)
class ArtifactRef:
    """Référence d'un artefact dans le manifeste de lignée (ADR-002).

    Attributes:
        path: chemin de l'artefact **relatif à la racine du dépôt** (ADR-002).
        kind: nature de l'artefact, une valeur de :data:`ARTIFACT_KINDS`.
        hashes: empreintes SHA-256 des fichiers produits, par nom de fichier
            (ADR-002 : empreintes de **fichier**, recalculables par un tiers).
    """

    path: str
    kind: str
    hashes: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not self.path.strip():
            raise ValueError("path doit être une chaîne non vide")
        if self.path.startswith("/") or (len(self.path) > 1 and self.path[1] == ":"):
            raise ValueError(f"path doit être relatif à la racine du dépôt, reçu {self.path!r}")
        if ".." in self.path.split("/"):
            raise ValueError(f"path ne doit pas remonter hors du dépôt, reçu {self.path!r}")
        if self.kind not in ARTIFACT_KINDS:
            raise ValueError(
                f"kind doit être l'un de {ARTIFACT_KINDS}, reçu {self.kind!r}"
            )
        if not isinstance(self.hashes, Mapping):
            raise TypeError("hashes doit être un mapping")
        for name, digest in self.hashes.items():
            if not isinstance(name, str) or not isinstance(digest, str):
                raise TypeError("hashes doit associer des noms de fichiers à des chaînes")

    def to_dict(self) -> dict:
        """Forme JSON du bloc ``artifact`` / ``parent``."""
        return {
            "path": self.path,
            "kind": self.kind,
            "hashes": dict(self.hashes),
        }


@dataclass(frozen=True)
class LineageManifest:
    """Manifeste de lignée ADR-002 produit par d-OPD.

    Le bloc ``vector`` du schéma ADR-002 est **omis** : il décrit τ, qui est
    propre à la fusion A2D. d-OPD ne fusionne rien (DOPD_SPEC §6.5) : les blocs
    qu'il renseigne sont ``artifact``, ``parent``, ``producer``, ``config``,
    ``data`` et ``verification``.

    Attributes:
        artifact: artefact produit par ce run.
        parent: source déclarée de l'artefact ; ``None`` pour un artefact
            racine (ADR-002 : ``"parent": null``).
        producer: bloc producteur ; ``module`` vaut toujours ``"dopd"``.
        config: hyperparamètres sérialisés (DOPDConfig, seed, etc.).
        data: description des données (empreintes des prompts, vérificateur).
        verification: contrôle effectué (smoke de chargement, évaluation, ...).
        schema: identifiant de schéma, figé par ADR-002.
    """

    artifact: ArtifactRef
    parent: ArtifactRef | None
    producer: Mapping[str, object]
    config: Mapping[str, object]
    data: Mapping[str, object]
    verification: Mapping[str, object]
    schema: str = LINEAGE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != LINEAGE_SCHEMA:
            raise ValueError(
                f"schema doit être {LINEAGE_SCHEMA!r}, reçu {self.schema!r}"
            )
        if not isinstance(self.artifact, ArtifactRef):
            raise TypeError("artifact doit être un ArtifactRef")
        if self.parent is not None and not isinstance(self.parent, ArtifactRef):
            raise TypeError("parent doit être un ArtifactRef ou None")
        if not isinstance(self.producer, Mapping):
            raise TypeError("producer doit être un mapping")
        if self.producer.get("module") != PRODUCER_MODULE:
            raise ValueError(
                "producer.module doit valoir "
                f"{PRODUCER_MODULE!r}, reçu {self.producer.get('module')!r}"
            )
        for name in ("config", "data", "verification"):
            if not isinstance(getattr(self, name), Mapping):
                raise TypeError(f"{name} doit être un mapping")

    def to_dict(self) -> dict:
        """Forme JSON complète du manifeste."""
        return {
            "schema": self.schema,
            "artifact": self.artifact.to_dict(),
            "parent": None if self.parent is None else self.parent.to_dict(),
            "producer": dict(self.producer),
            "config": dict(self.config),
            "data": dict(self.data),
            "verification": dict(self.verification),
        }


# ---------------------------------------------------------------------------
# Contrats
# ---------------------------------------------------------------------------


class TeacherScorer(ABC):
    """Contrat du professeur AR gelé : prior causal ``r_i(v)`` et score ``s_i(v)``.

    Le professeur joue deux rôles dans d-OPD (DOPD_SPEC §6.3) : il fournit le
    prior causal ``r_i(v)`` (eq. 11) **et** le score de compatibilité du futur
    ``s_i(v)`` (eq. 14). Les deux requêtes partagent le même préfixe causal
    ``(q, z, Â_i)``, qui est donc mis en cache une seule fois (App. B).

    L'implémentation concrète vit **hors** du paquet pur (Hugging Face /
    Qwen3-0.6B) : ce contrat ne connaît ni tokenizer ni checkpoint.

    Références :
      - arXiv:2609.35362 §4.1-4.2, App. B ;
      - DOPD_SPEC §3.2 (eq. 11, eq. 14), §3.3 (eq. 16-18), §6.2, §6.3.
    """

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Taille du vocabulaire du professeur (indexation des candidats)."""

    @abstractmethod
    def causal_prior(self, prefix_ids: Sequence[int]) -> np.ndarray:
        """Prior causal ``r_i(v) = p_T(v | q, z, Â_i)`` (eq. 11).

        Rend la distribution sur **tout** le vocabulaire, de somme 1 : c'est la
        base de la renormalisation plein vocabulaire de la cible (eq. 18).
        """

    @abstractmethod
    def suffix_loglikelihood(
        self,
        prefix_ids: Sequence[int],
        candidate_ids: Sequence[int],
        suffix_ids: Sequence[int],
    ) -> np.ndarray:
        """Score de suffixe ``s_i(v) = log p_T(Ĉ_i | q, z, Â_i, v)`` (eq. 14).

        Une branche par candidat : ``candidate_ids`` liste les tokens à insérer
        à la position ``i`` et le rendu est aligné dessus. Le préfixe causal
        ``prefix_ids`` est commun à toutes les branches et se met en cache
        (App. B) ; ``suffix_ids`` est le suffixe complété ``Ĉ_i``.
        """

    def score(self, request: ScoringRequest) -> ScoringResult:
        """Exécute une requête complète et rend le résultat exploitable.

        Appelle :meth:`causal_prior` une fois, puis :meth:`suffix_loglikelihood`
        une fois en demandant **toutes** les branches d'un seul lot : les
        candidats de ``V_i`` plus la branche de référence ``x̂_i`` (au plus
        ``2k + 1 = 33`` branches à ``k = 16``, App. B). Si ``x̂_i`` fait déjà
        partie de ``V_i``, il n'est pas dupliqué.
        """
        vocab_size = int(self.vocab_size)
        if vocab_size <= 0:
            raise ValueError(f"vocab_size doit être > 0, reçu {vocab_size}")

        prior = np.asarray(self.causal_prior(request.prefix_ids), dtype=np.float64)
        if prior.shape != (vocab_size,):
            raise ValueError(
                "causal_prior doit rendre un vecteur de taille vocab_size "
                f"({prior.shape} != ({vocab_size},))"
            )

        branches = list(request.candidate_ids)
        if request.reference_id not in branches:
            branches.append(request.reference_id)
        scores = np.asarray(
            self.suffix_loglikelihood(
                request.prefix_ids, tuple(branches), request.suffix_ids
            ),
            dtype=np.float64,
        )
        if scores.shape != (len(branches),):
            raise ValueError(
                "suffix_loglikelihood doit rendre un vecteur aligné sur "
                f"candidate_ids ({scores.shape} != ({len(branches)},))"
            )

        lookup = {token: float(score) for token, score in zip(branches, scores)}
        candidate_scores = np.array(
            [lookup[token] for token in request.candidate_ids], dtype=np.float64
        )
        return ScoringResult(
            prior=prior,
            candidate_ids=request.candidate_ids,
            candidate_loglikelihoods=candidate_scores,
            reference_loglikelihood=lookup[request.reference_id],
        )


class RolloutRecorder(ABC):
    """Contrat d'enregistrement du rollout de l'élève.

    Le rollout de d-OPD produit deux choses (App. A, étapes 1 et 2) : les
    **états de débruitage visités** ``X_t`` et le **bloc complété** ``x̂``. Le
    premier sert à choisir l'état sur lequel la cible est calculée, le second
    fournit le préfixe ``Â_i``, le suffixe ``Ĉ_i`` et le token de référence
    ``x̂_i``.

    Références :
      - arXiv:2609.35362 App. A (algorithme 1, étapes 1-3), App. C.2 (règle
        d'échantillonnage) ;
      - DOPD_SPEC §2.1 (notation), §3.5 (étapes 1-3), §6.2.
    """

    @abstractmethod
    def record_state(self, state: DenoisingState) -> None:
        """Enregistre un état de débruitage visité ``X_t`` (eq. 2)."""

    @abstractmethod
    def record_block(self, block: CompletedBlock) -> None:
        """Enregistre un bloc complété ``x̂`` (eq. 4)."""

    @abstractmethod
    def states(self, block_index: int | None = None) -> tuple[DenoisingState, ...]:
        """États visités, pour un bloc donné ou pour toute la trajectoire."""

    @abstractmethod
    def blocks(self) -> tuple[CompletedBlock, ...]:
        """Blocs complétés de la trajectoire, dans l'ordre d'enregistrement."""

    def sample_state(
        self, block_index: int, rng: np.random.Generator
    ) -> DenoisingState:
        """Tire **un** état de débruitage pour le bloc ``block_index``.

        Règle du papier (App. C.2) : un seul état est échantillonné par bloc
        généré, uniformément parmi les tours visités où le bloc a **au moins
        une position masquée** — un état entièrement visible n'a rien à
        corriger.
        """
        eligible = [
            state for state in self.states(block_index) if state.masked_positions
        ]
        if not eligible:
            raise LookupError(
                f"aucun état avec position masquée pour le bloc {block_index}"
            )
        return eligible[int(rng.integers(len(eligible)))]


class TaskVerifier(ABC):
    """Contrat du vérificateur de tâche : l'étiquette de correction ``R(x̂)``.

    La correction future-aware n'est appliquée que si le rollout complété passe
    le vérificateur ; sinon le rollout reçoit la cible causale standard et
    reste dans l'entraînement (DOPD_SPEC §3.4). Le papier mesure l'effet de la
    porte à 58,8 → 59,8 de moyenne six benchmarks (8B, ``N = 4``, Table 3).

    Références :
      - arXiv:2609.35362 §4.2, Table 3 ;
      - DOPD_SPEC §3.4 (porte), §3.5 (étape 2), §6.2.
    """

    @abstractmethod
    def verify(self, completion: str) -> bool:
        """Vérifie le rollout complété et rend l'étiquette ``R(x̂)``."""

    def gate_open(self, completion: str, *, config: DOPDConfig) -> bool:
        """Indique si la correction future-aware est applicable.

        La porte est ouverte quand la configuration l'active
        (``correctness_gate``) **et** que le vérificateur accepte le rollout.
        Porte désactivée, la correction s'applique inconditionnellement : c'est
        le mode de repli documenté quand aucun vérificateur n'est disponible.
        """
        if not config.correctness_gate:
            return True
        return bool(self.verify(completion))

    def correction_label(self, completion: str, *, config: DOPDConfig) -> int:
        """Étiquette de correction ``R(x̂)`` : 1 si la porte est ouverte, 0 sinon."""
        return 1 if self.gate_open(completion, config=config) else 0


class LineageWriter(ABC):
    """Contrat d'écriture du manifeste de lignée ADR-002.

    d-OPD n'écrit pas de ``merge_config.json`` (ce n'est pas une fusion) : tout
    artefact produit — adaptateur, checkpoint d'élève, résultat d'évaluation —
    porte un ``lineage.json`` écrit **à côté** de lui, au schéma
    ``academiia.lineage/1``, avec ``producer.module = "dopd"``.

    ADR-002 assigne l'écriture/lecture générique du manifeste au module
    ``core``, qui ne l'implémente pas encore : ce contrat fournit donc la
    **construction** du manifeste (pure) et laisse l'écriture sur disque à une
    implémentation externe, pour ne pas créer de format concurrent.

    Références :
      - `docs/adr/ADR-002-provenance.md` ;
      - DOPD_SPEC §6.5 (champs d-OPD), §6.2 (interface).
    """

    @abstractmethod
    def write(self, manifest: LineageManifest) -> str:
        """Écrit le manifeste et rend le chemin du fichier écrit."""

    @staticmethod
    def config_block(config: DOPDConfig, **extra: object) -> dict:
        """Bloc ``config`` du manifeste : ``DOPDConfig`` sérialisée + extras.

        Contient par défaut la taille de bloc, ``k``, la porte et les dtypes
        (DOPD_SPEC §6.5) ; ``extra`` permet d'y ajouter la seed ou toute
        information propre au run.
        """
        block = config.to_dict()
        block.update(extra)
        return block

    def build_manifest(
        self,
        *,
        artifact: ArtifactRef,
        parent: ArtifactRef | None,
        config: Mapping[str, object],
        data: Mapping[str, object],
        verification: Mapping[str, object],
        command: str,
    ) -> LineageManifest:
        """Construit le manifeste ADR-002 avec ``producer.module = "dopd"``.

        Le bloc ``producer`` porte la version du paquet, le commit git courant
        (helper ``academiia.core.git_commit``) et la commande exacte du run.
        """
        if not isinstance(command, str) or not command.strip():
            raise ValueError("command doit être une chaîne non vide")
        return LineageManifest(
            artifact=artifact,
            parent=parent,
            producer={
                "module": PRODUCER_MODULE,
                "version": PACKAGE_VERSION,
                "commit": git_commit(),
                "command": command,
            },
            config=dict(config),
            data=dict(data),
            verification=dict(verification),
        )
