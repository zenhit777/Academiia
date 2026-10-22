"""Mapping de noms Gemma 4 ↔ DiffusionGemma et helpers de τ legacy.

Ces fonctions sont **spécifiques A2D** (elles connaissent les architectures du
papier) : elles ne vivent donc pas dans ``academiia.core``. Les helpers
génériques (empreintes, sidecars, commit git) sont dans
``academiia.core.common``.
"""

from __future__ import annotations

import json
import os


def gemma4_to_diffusiongemma(name: str):
    """Gemma 4 (A) -> DiffusionGemma (D) decoder. None si pas d'equivalent."""
    if name.startswith("model.language_model.layers."):
        return "model.decoder.layers." + name[len("model.language_model.layers."):]
    if name == "model.language_model.embed_tokens.weight":
        return "model.decoder.embed_tokens.weight"
    return None


def diffusiongemma_to_gemma4(dg_name: str):
    """DiffusionGemma decoder -> Gemma 4. None si pas d'equivalent."""
    if dg_name.startswith("model.decoder.layers."):
        return "model.language_model.layers." + dg_name[len("model.decoder.layers."):]
    if dg_name == "model.decoder.embed_tokens.weight":
        return "model.language_model.embed_tokens.weight"
    return None


def is_norm_or_embed(dg_name: str) -> bool:
    n = dg_name.lower()
    return (
        "layernorm" in n
        or "norm" in n
        or "embed_tokens" in n
        or n.endswith("layer_scalar")
    )


def family_of(dg_name: str) -> str:
    """Famille d'un tenseur DiffusionGemma (agregation des mesures/rapports).

    Valeurs : 'embed', 'norms', 'router', 'attn', 'mlp/experts', 'autre'.
    Les normes (y compris q_norm/k_norm/layer_scalar) sont classees AVANT
    'attn'/'mlp' pour rester coherentes avec is_norm_or_embed().
    """
    n = dg_name.lower()
    if "embed" in n:
        return "embed"
    if "layernorm" in n or "norm" in n or n.endswith("layer_scalar"):
        return "norms"
    if "router" in n:
        return "router"
    if "self_attn" in n:
        return "attn"
    if "mlp" in n or "experts" in n or "moe" in n:
        return "mlp/experts"
    return "autre"


def tau_filename(g4_name: str) -> str:
    return g4_name.replace(".", "_") + ".npy"


def check_tau_manifest(tau_dir: str, allow_missing: bool = False):
    """Valide tau_manifest.json contre les .npy presents dans tau_dir.

    Retourne (manifest|None, erreur|None). Une erreur non nulle doit etre
    traitee comme un echec (exit 2) par l'appelant.
    """
    path = os.path.join(tau_dir, "tau_manifest.json")
    npy = sorted(f for f in os.listdir(tau_dir) if f.endswith(".npy"))
    if not os.path.exists(path):
        if allow_missing:
            return None, None
        return None, (
            f"{path} absent : provenance du tau non verifiable. Recalculer tau "
            f"avec modules/a2d/legacy/scripts/compute_tau_a.py, ou passer "
            f"--allow-missing-tau-manifest."
        )
    with open(path) as f:
        manifest = json.load(f)
    declared = manifest.get("n_tensors")
    if declared is not None and declared != len(npy):
        return manifest, (
            f"manifest incoherent : n_tensors={declared} mais {len(npy)} "
            f"fichiers .npy presents dans {tau_dir}"
        )
    filt = manifest.get("filter")
    if filt not in (None, "all", "decoder"):
        return manifest, (
            f"manifest : filter='{filt}' invalide (attendu 'all' ou 'decoder')"
        )
    return manifest, None
