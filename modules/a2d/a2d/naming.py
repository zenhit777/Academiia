"""Nommage des tenseurs : familles et mappings de noms AR -> diffusion.

Le mapping est *pluggable* : un `NameMapping` sait transformer un chemin de
module source (AR) en chemin de module cible (diffusion) et l'inverse, et
declare quelles familles de tenseurs sont fusionnables.

Les chemins de module sont les noms de tenseurs sans le suffixe ``.weight``
(``model.layers.0.self_attn.q_proj.weight`` -> ``model.layers.0.self_attn.q_proj``).

References :
  - A2D App. A.2 (PDF p. 19) : « For Dream-7B, Dream-Coder, DiffuCoder, and
    DreamReasoner the diffusion model keeps its AR parent's architecture,
    parameter names, and tensor shapes, so the merge is a direct element-wise
    combination of all parameters, with no remapping, freezing, or masking. »
  - A2D App. A.2 (PDF p. 19) : « The vision encoder, embeddings, normalization
    parameters, and output head are unchanged » (DiffusionGemma / Gemma 4).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, FrozenSet, Optional

from .mapping import (diffusiongemma_to_gemma4,  # mapping officiel du projet
                      gemma4_to_diffusiongemma)

FAMILY_EMBED = "embed"
FAMILY_HEAD = "head"
FAMILY_NORMS = "norms"
FAMILY_ROUTER = "router"
FAMILY_EXPERTS = "experts"
FAMILY_ATTN = "attn"
FAMILY_MLP = "mlp"
FAMILY_OTHER = "other"

ALL_FAMILIES = (FAMILY_EMBED, FAMILY_HEAD, FAMILY_NORMS, FAMILY_ROUTER,
                FAMILY_EXPERTS, FAMILY_ATTN, FAMILY_MLP, FAMILY_OTHER)


def family_of(name: str) -> str:
    """Famille d'un tenseur, a partir du nom (avec ou sans ``.weight``).

    L'ordre des tests est significatif : ``embed`` avant ``norms`` avant
    ``attn``/``mlp`` (un ``q_norm`` est une norme, pas une attention).
    """
    n = name.lower()
    if "embed" in n:
        return FAMILY_EMBED
    if "lm_head" in n or n.endswith("head.weight"):
        return FAMILY_HEAD
    if "router" in n:
        return FAMILY_ROUTER
    if "experts" in n:
        return FAMILY_EXPERTS
    if "norm" in n or n.endswith("layer_scalar"):
        return FAMILY_NORMS
    if "self_attn" in n or "attn" in n:
        return FAMILY_ATTN
    if "mlp" in n or "feed_forward" in n:
        return FAMILY_MLP
    return FAMILY_OTHER


def module_of(tensor_name: str) -> str:
    """Chemin de module associe a un nom de tenseur."""
    if tensor_name.endswith(".weight"):
        return tensor_name[: -len(".weight")]
    return tensor_name


@dataclass(frozen=True)
class NameMapping:
    """Mapping de noms entre l'espace AR (source) et l'espace diffusion (cible).

    ``rename`` : module source -> module cible (None si pas d'equivalent).
    ``inverse`` : module cible -> module source (None si pas d'equivalent).
    ``fusible_families`` : familles de tenseurs fusionnees par le merge.
    ``router_policy`` : ``merge`` (defaut, comme le merge historique) ou
        ``skip`` ; le routeur est une famille a part, declaree explicitement.
    ``embedding_policy`` : ``rows`` (intersection des lignes si le
        vocabulaire differe, tau=0 sur les lignes en trop, journalise) ou
        ``strict`` (erreur sur forme differente).
    """

    id: str
    description: str
    rename: Callable[[str], Optional[str]]
    inverse: Callable[[str], Optional[str]]
    fusible_families: FrozenSet[str]
    router_policy: str = "merge"
    embedding_policy: str = "rows"

    def target(self, source_module: str) -> Optional[str]:
        return self.rename(source_module)

    def source(self, target_module: str) -> Optional[str]:
        return self.inverse(target_module)

    def is_fusable(self, family: str) -> bool:
        if family == FAMILY_ROUTER and self.router_policy == "skip":
            return False
        return family in self.fusible_families

    def describe(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "fusible_families": sorted(self.fusible_families),
            "router_policy": self.router_policy,
            "embedding_policy": self.embedding_policy,
        }


def _identity(name: str) -> Optional[str]:
    return name


IDENTITY = NameMapping(
    id="identity",
    description=("Qwen2.5-7B -> Dream-7B : le modele de diffusion conserve "
                 "l'architecture, les noms et les formes de son parent AR "
                 "(A2D App. A.2) ; fusion de TOUS les parametres, embeddings "
                 "et tete de sortie inclus."),
    rename=_identity,
    inverse=_identity,
    fusible_families=frozenset(ALL_FAMILIES),
    router_policy="merge",
    embedding_policy="rows",
)


def _gemma4_module_to_diffusiongemma(module: str) -> Optional[str]:
    """Chemin de module Gemma 4 -> chemin de module DiffusionGemma."""
    mapped = gemma4_to_diffusiongemma(module + ".weight")
    if mapped is None:
        return None
    return module_of(mapped)


def _diffusiongemma_module_to_gemma4(module: str) -> Optional[str]:
    """Chemin de module DiffusionGemma -> chemin de module Gemma 4."""
    mapped = diffusiongemma_to_gemma4(module + ".weight")
    if mapped is None:
        return None
    return module_of(mapped)


GEMMA4_TO_DIFFUSIONGEMMA = NameMapping(
    id="gemma4-diffusiongemma",
    description=("Gemma 4 -> DiffusionGemma : mapping par nom des projections "
                 "du decoder ; vision, embeddings, normalisations et tete de "
                 "sortie inchanges (A2D App. A.2)."),
    rename=_gemma4_module_to_diffusiongemma,
    inverse=_diffusiongemma_module_to_gemma4,
    fusible_families=frozenset({FAMILY_ATTN, FAMILY_MLP, FAMILY_EXPERTS, FAMILY_ROUTER}),
    router_policy="merge",
    embedding_policy="strict",
)

MAPPINGS = {IDENTITY.id: IDENTITY, GEMMA4_TO_DIFFUSIONGEMMA.id: GEMMA4_TO_DIFFUSIONGEMMA}


def get_mapping(mapping_id: str, router_policy: str = "merge") -> NameMapping:
    """Retourne le mapping demande ; ``router_policy`` est applique."""
    if mapping_id not in MAPPINGS:
        raise KeyError(f"mapping inconnu : {mapping_id!r} (dispo : {sorted(MAPPINGS)})")
    base = MAPPINGS[mapping_id]
    if router_policy == base.router_policy:
        return base
    return NameMapping(
        id=base.id,
        description=base.description,
        rename=base.rename,
        inverse=base.inverse,
        fusible_families=base.fusible_families,
        router_policy=router_policy,
        embedding_policy=base.embedding_policy,
    )
