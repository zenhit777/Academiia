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


def tau_filename(g4_name: str) -> str:
    return g4_name.replace(".", "_") + ".npy"


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
    """Tous les fichiers source a recopier : tout sauf poids et index."""
    out = []
    for entry in sorted(os.listdir(src_dir)):
        if entry.endswith(".safetensors"):
            continue
        if entry == "model.safetensors.index.json":
            continue
        if entry.startswith("."):
            continue
        out.append(entry)
    return out
