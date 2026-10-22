"""Empreintes, index HuggingFace et fichiers annexes — helpers génériques.

Aucune dépendance vers un module : ce fichier ne connaît ni A2D ni les
conventions de nommage d'un modèle. Lecture des safetensors via
``safetensors.safe_open`` uniquement (jamais de parseur maison).
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess


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
    """Commit court du dépôt courant ("unknown" hors dépôt)."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def index_fingerprint(index_path: str) -> dict:
    """Empreinte légère de l'index : nb tenseurs + hash du weight_map trié."""
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
    """Fichiers source à recopier : tout sauf poids, index, README et dotfiles.

    README.md est EXCLU de la recopie : la model card de la destination ne doit
    pas être remplacée par celle de la source. Le wrapper legacy ne copie le
    README source que si la destination n'en a pas ; ``verify_checkpoint.py
    --reference`` le compare à part.
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
