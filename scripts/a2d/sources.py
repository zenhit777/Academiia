"""Sources de vecteurs de tache (tau).

Trois sources, toutes en lecture seule :

* `CheckpointPairSource` : tau = W(post-trained) - W(base), calcule a la volee
  depuis deux checkpoints complets (3-4 handles `safe_open` ouverts en
  permanence, un seul cast FP32 par tenseur).
* `LoraAdapterSource` : tau = (lora_alpha / r) * B @ A, par module adapte
  (identique a `peft.merge_and_unload` sur le modele de base).
* `LegacyNpySource` : lecture des `.npy` historiques produits par
  `legacy/scripts/compute_tau_a.py`. Reserve au wrapper deprecie
  `legacy/scripts/merge_jainius.py` ; le pipeline A2D conforme ne persiste JAMAIS
  de tau sur disque.

Lecture des safetensors : EXCLUSIVEMENT via `safetensors.safe_open`
(bits exacts). Aucun parseur maison (`.cursorrules`).
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from safetensors import safe_open

from .naming import module_of


class VectorSource(ABC):
    """Fournit un vecteur de tache (delta) par chemin de module."""

    kind = "abstract"
    enumerable = True

    @abstractmethod
    def has(self, module: str) -> bool:
        """Le module existe-t-il dans cette source ?"""

    @abstractmethod
    def delta(self, module: str) -> torch.Tensor:
        """Delta FP32 du module donne (module sans suffixe `.weight`)."""

    def modules(self):
        """Modules disponibles (None si la source n'est pas enumerable)."""
        return None

    def describe(self) -> dict:
        raise NotImplementedError

    def close(self) -> None:
        """Libere les ressources (handles safe_open)."""


def load_weight_map(directory) -> Dict[str, str]:
    """weight_map {nom_tenseur: fichier} d'un checkpoint (index ou mono-fichier)."""
    directory = Path(directory)
    index_path = directory / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            return json.load(f)["weight_map"]
    single = directory / "model.safetensors"
    if single.exists():
        with safe_open(str(single), framework="pt") as f:
            keys = list(f.keys())
        return {k: "model.safetensors" for k in keys}
    raise FileNotFoundError(
        f"index safetensors introuvable dans {directory} "
        f"(model.safetensors.index.json ou model.safetensors)")


def index_fingerprint(directory) -> dict:
    """Empreinte de l'index d'un checkpoint (provenance, pas de cache)."""
    directory = Path(directory)
    index_path = directory / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            wm = json.load(f)["weight_map"]
        import hashlib
        canonical = json.dumps(wm, sort_keys=True, separators=(",", ":"))
        return {
            "n_tensors": len(wm),
            "n_files": len(set(wm.values())),
            "weight_map_sha256": hashlib.sha256(canonical.encode()).hexdigest()[:32],
        }
    return {"n_tensors": len(load_weight_map(directory)), "n_files": 1,
            "weight_map_sha256": "mono-fichier"}


def shard_fingerprints(directory, weight_map, limit_mb: int = 32) -> dict:
    """SHA256 (prefixe de `limit_mb` Mo) de chaque shard : preuve de contenu.

    L'empreinte d'index ne distingue pas deux checkpoints de meme architecture
    (Qwen2.5-7B et Qwen2.5-7B-Instruct ont le meme weight_map) : le prefixe de
    contenu des shards identifie reellement les poids utilises.
    """
    from _common import sha256_file
    out = {}
    for fname in sorted(set(weight_map.values())):
        path = os.path.join(str(directory), fname)
        out[fname] = sha256_file(path, limit_mb=limit_mb)
    return out


class _SafeOpenPool:
    """Handles `safe_open` reutilises (un par fichier, jamais plus de 4)."""

    def __init__(self) -> None:
        self._handles: Dict[str, object] = {}

    def get(self, path: str):
        handle = self._handles.get(path)
        if handle is None:
            handle = safe_open(path, framework="pt")
            self._handles[path] = handle
        return handle

    def close(self) -> None:
        for handle in self._handles.values():
            close = getattr(handle, "__exit__", None)
            if close is not None:
                try:
                    close(None, None, None)
                except Exception:
                    pass
        self._handles.clear()


class CheckpointPairSource(VectorSource):
    """tau = W(post-trained) - W(base), calcule tenseur par tenseur en FP32."""

    kind = "checkpoint-pair"

    def __init__(self, post_trained, base) -> None:
        self.post_trained = Path(post_trained)
        self.base = Path(base)
        self.post_map = load_weight_map(self.post_trained)
        self.base_map = load_weight_map(self.base)
        self._post_module = {module_of(t): t for t in self.post_map}
        self._base_module = {module_of(t): t for t in self.base_map}
        self._common = {m for m in self._post_module if m in self._base_module}
        self._pool = _SafeOpenPool()

    def has(self, module: str) -> bool:
        return module in self._common

    def modules(self):
        return set(self._common)

    def _tensor(self, directory: Path, weight_map, module_map, module: str) -> torch.Tensor:
        tensor_name = module_map[module]
        handle = self._pool.get(str(directory / weight_map[tensor_name]))
        return handle.get_tensor(tensor_name).to(torch.float32)

    def delta(self, module: str) -> torch.Tensor:
        post = self._tensor(self.post_trained, self.post_map, self._post_module, module)
        base = self._tensor(self.base, self.base_map, self._base_module, module)
        return post - base

    def tensor(self, module: str, side: str = "post") -> torch.Tensor:
        """Tenseur brut (FP32) d'un module : side 'post' (A_p/D_p) ou 'base' (A_b/D_b)."""
        if side == "post":
            return self._tensor(self.post_trained, self.post_map, self._post_module, module)
        if side == "base":
            return self._tensor(self.base, self.base_map, self._base_module, module)
        raise ValueError(f"side inconnu : {side!r} (attendu 'post' ou 'base')")

    def describe(self) -> dict:
        return {
            "kind": self.kind,
            "post_trained": str(self.post_trained),
            "base": str(self.base),
            "n_tensors_common": len(self._common),
            "index_post_trained": index_fingerprint(self.post_trained),
            "index_base": index_fingerprint(self.base),
            "shards_post_trained": shard_fingerprints(self.post_trained, self.post_map),
            "shards_base": shard_fingerprints(self.base, self.base_map),
        }

    def close(self) -> None:
        self._pool.close()


class LoraAdapterSource(VectorSource):
    """tau = (lora_alpha / r) * B @ A, par module adapte.

    Le facteur `lora_alpha / r` est celui de PEFT (`use_rslora` : alpha/sqrt(r)).
    Les cles de l'adaptateur sont prefixees par `base_model.model.` (PEFT).
    """

    kind = "lora-adapter"
    PEFT_PREFIX = "base_model.model."

    def __init__(self, adapter_dir) -> None:
        self.dir = Path(adapter_dir)
        config_path = self.dir / "adapter_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"adapter_config.json introuvable dans {self.dir}")
        with open(config_path) as f:
            self.config = json.load(f)
        self.r = int(self.config["r"])
        self.alpha = float(self.config["lora_alpha"])
        if self.config.get("use_rslora"):
            self.scale = self.alpha / (self.r ** 0.5)
        else:
            self.scale = self.alpha / self.r
        self.target_modules = self.config.get("target_modules")
        self._files = self._locate_weights()
        self._pool = _SafeOpenPool()
        self._modules = self._index_modules()

    def _locate_weights(self):
        index_path = self.dir / "model.safetensors.index.json"
        if index_path.exists():
            with open(index_path) as f:
                return json.load(f)["weight_map"]
        single = self.dir / "adapter_model.safetensors"
        if not single.exists():
            raise FileNotFoundError(f"adapter_model.safetensors introuvable dans {self.dir}")
        with safe_open(str(single), framework="pt") as f:
            return {k: "adapter_model.safetensors" for k in f.keys()}

    def _index_modules(self) -> Dict[str, Dict[str, str]]:
        modules: Dict[str, Dict[str, str]] = {}
        for tensor_name in self._files:
            for suffix, slot in ((".lora_A.weight", "A"), (".lora_B.weight", "B")):
                if tensor_name.endswith(suffix):
                    base = tensor_name[: -len(suffix)]
                    if base.startswith(self.PEFT_PREFIX):
                        base = base[len(self.PEFT_PREFIX):]
                    modules.setdefault(base, {})[slot] = tensor_name
                    break
        return modules

    def has(self, module: str) -> bool:
        entry = self._modules.get(module)
        return bool(entry) and "A" in entry and "B" in entry

    def modules(self):
        return {m for m, e in self._modules.items() if "A" in e and "B" in e}

    def delta(self, module: str) -> torch.Tensor:
        entry = self._modules[module]
        a_name, b_name = entry["A"], entry["B"]
        a = self._pool.get(str(self.dir / self._files[a_name])).get_tensor(a_name).to(torch.float32)
        b = self._pool.get(str(self.dir / self._files[b_name])).get_tensor(b_name).to(torch.float32)
        return self.scale * (b @ a)

    def describe(self) -> dict:
        return {
            "kind": self.kind,
            "adapter_dir": str(self.dir),
            "r": self.r,
            "lora_alpha": self.alpha,
            "scale": self.scale,
            "use_rslora": bool(self.config.get("use_rslora")),
            "target_modules": self.target_modules,
            "n_modules": len(self._modules),
            "shards_sha256": shard_fingerprints(self.dir, self._files, limit_mb=64),
        }

    def close(self) -> None:
        self._pool.close()


class LegacyNpySource(VectorSource):
    """Lecture des `.npy` historiques (legacy/scripts/compute_tau_a.py).

    LEGACY : n'est utilise que par le wrapper deprecie `legacy/scripts/merge_jainius.py`.
    Le pipeline A2D conforme calcule tau a la volee et ne persiste rien.
    """

    kind = "legacy-npy"
    enumerable = False

    def __init__(self, tau_dir) -> None:
        self.dir = Path(tau_dir)
        self.manifest = None
        manifest_path = self.dir / "tau_manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                self.manifest = json.load(f)

    @staticmethod
    def filename(module: str) -> str:
        """Nom de fichier legacy (compute_tau_a.py) : nom du TENSEUR, pas du module."""
        return (module + ".weight").replace(".", "_") + ".npy"

    @staticmethod
    def _candidates(module: str):
        return [LegacyNpySource.filename(module),
                module.replace(".", "_") + ".npy"]

    def has(self, module: str) -> bool:
        return any((self.dir / name).exists() for name in self._candidates(module))

    def _path(self, module: str):
        for name in self._candidates(module):
            path = self.dir / name
            if path.exists():
                return path
        raise FileNotFoundError(f"tau legacy introuvable pour {module} dans {self.dir}")

    def delta(self, module: str) -> torch.Tensor:
        path = self._path(module)
        return torch.from_numpy(np.load(path).astype(np.float32, copy=False))

    def describe(self) -> dict:
        return {
            "kind": self.kind,
            "tau_dir": str(self.dir),
            "tau_manifest": self.manifest,
            "n_npy": len([f for f in os.listdir(self.dir) if f.endswith(".npy")]),
        }
