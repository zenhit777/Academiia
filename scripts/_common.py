"""Helpers partages du pipeline Dp + alpha*tau_A (sans parseur maison).

Lire les safetensors EXCLUSIVEMENT via `safetensors.safe_open`
(bits exacts). Ne pas reintroduire de parseur struct/json manuel.
"""

import hashlib
import json
import os
import subprocess


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
            f"avec compute_tau_a.py, ou passer --allow-missing-tau-manifest."
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


def sha256_file(path: str, limit_mb: int = 0) -> str:
    """SHA256 d'un fichier (limit_mb=0 -> entier)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(64 * 1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
            if limit_mb and f.tell() > limit_mb * 1024 * 1024:
                break
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def index_fingerprint(index_path: str) -> dict:
    """Empreinte legere de l'index : nb tenseurs + hash du weight_map trie."""
    with open(index_path) as f:
        idx = json.load(f)
    wm = idx.get("weight_map", {})
    canonical = json.dumps(wm, sort_keys=True, separators=(",", ":"))
    return {
        "n_tensors": len(wm),
        "n_files": len(set(wm.values())),
        "weight_map_sha256": hashlib.sha256(canonical.encode()).hexdigest()[:16],
    }


def list_source_sidecar_files(src_dir: str) -> list:
    """Fichiers source a recopier : tout sauf poids, index, README et dotfiles.

    README.md est EXCLU de la recopie : la model card de la destination (Jainius)
    ne doit pas etre remplacee par celle de DiffusionGemma. merge_jainius.py ne
    copie le README source que si la destination n'en a pas (et
    verify_checkpoint.py --reference le compare a part).
    """
    out = []
    for entry in sorted(os.listdir(src_dir)):
        if entry.endswith(".safetensors"):
            continue
        if entry == "model.safetensors.index.json":
            continue
        if entry == "README.md":
            continue
        if entry.startswith("."):
            continue
        out.append(entry)
    return out
