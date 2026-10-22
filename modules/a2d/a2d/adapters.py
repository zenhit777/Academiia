"""Mise a l'echelle d'adaptateurs LoRA (mode `merge-dg-scaled`, App. A.2).

Pour DiffusionGemma, l'adaptateur de diffusion publie reste **NON fusionne**
dans les poids (App. A.2 : « *stay unmerged and are scaled* »). Le mode
`merge-dg-scaled` ecrit donc deux artefacts :

* un checkpoint de base modifie `base' = D_b + lambda*tau_A` (writer A2D) ;
* une copie de l'adaptateur dont les tenseurs `lora_B` sont multiplies par
  `(1 - lambda)` (option (b) du design `docs/design/G2_adapter_scaling.md`).

L'echelle PEFT etant `lora_alpha / r`, multiplier `B` par `s` multiplie le
delta `(alpha/r) * B @ A` par `s` : c'est exact en FP32, a l'arrondi de
l'ecriture de `B` pres. `lora_alpha` reste **intact** dans
`adapter_config.json` (aucune dependance a un arrondi de `alpha`).

La verification (option (c) du design) recharge l'adaptateur produit et
compare, module par module, `delta_scaled` a `factor * delta_original`.

Ecriture atomique (dossier temporaire + `os.replace`) et metadonnees
safetensors (`format: pt`) reprises de la source, comme pour les checkpoints.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .sources import LoraAdapterSource, shard_fingerprints

#: Dtypes d'ecriture acceptes pour l'adaptateur mis a l'echelle.
ADAPTER_DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32}

#: Suffixe des tenseurs LoRA multiplies par le facteur.
SCALE_SUFFIX = ".lora_B.weight"

#: Criteres de verification V5 (design G2, §5.1). L'ecriture en BF16 ajoute un
#: arrondi sur `lora_B` (epsilon BF16 = 2^-8 ~ 3,9e-3 par element) : le seuil
#: relatif est donc 1e-2 en BF16 et 1e-3 en FP32. Une erreur de facteur (ex.
#: 1,0 au lieu de 0,7) produit >= 40 % d'ecart : ces seuils restent largement
#: discriminants.
MAX_RELATIVE_L2_ERROR_BF16 = 1e-2
MAX_RELATIVE_L2_ERROR_FP32 = 1e-3
MAX_COSINE_DISTANCE = 1e-4

__all__ = ["ADAPTER_DTYPES", "AdapterError", "scale_adapter",
           "verify_scaled_adapter"]


class AdapterError(RuntimeError):
    """Erreur de mise a l'echelle / de verification d'adaptateur."""


def _dtype_of(name: str) -> torch.dtype:
    try:
        return ADAPTER_DTYPES[name]
    except KeyError:
        raise AdapterError(
            f"dtype d'adaptateur inconnu : {name!r} "
            f"(attendu : {sorted(ADAPTER_DTYPES)})") from None


def _atomic_save(tensors: Dict[str, torch.Tensor], path: Path,
                 metadata: Dict[str, str] | None = None) -> None:
    """Ecrit un shard safetensors atomiquement (metadonnees reprises)."""
    meta = dict(metadata) if metadata else {}
    if "format" not in meta:
        meta["format"] = "pt"
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".tmp-")
    os.close(fd)
    try:
        save_file(tensors, tmp_path, metadata=meta)
        os.replace(tmp_path, str(path))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _atomic_write_json(path: Path, payload: dict) -> None:
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".tmp-")
    os.close(fd)
    try:
        with open(tmp_path, "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, str(path))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _adapter_layout(directory: Path) -> Tuple[Dict[str, str], bool]:
    """(weight_map, sharded) de l'adaptateur ; refuse les formats non safetensors."""
    index_path = directory / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            return json.load(f)["weight_map"], True
    single = directory / "adapter_model.safetensors"
    if single.exists():
        with safe_open(str(single), framework="pt") as f:
            return {k: "adapter_model.safetensors" for k in f.keys()}, False
    if (directory / "adapter_model.bin").exists():
        raise AdapterError(
            f"{directory} ne contient qu'un adaptateur PyTorch (.bin) : le mode "
            "merge-dg-scaled exige des poids safetensors (lecture via safe_open)")
    raise AdapterError(
        f"poids d'adaptateur introuvables dans {directory} "
        "(adapter_model.safetensors ou model.safetensors.index.json)")


def _b_weight_dtype(directory) -> torch.dtype | None:
    """Dtype du premier tenseur `lora_B` (pour choisir la tolerance V5)."""
    weight_map, _ = _adapter_layout(Path(directory))
    for tensor_name, fname in weight_map.items():
        if tensor_name.endswith(SCALE_SUFFIX):
            with safe_open(str(Path(directory) / fname), framework="pt") as handle:
                return handle.get_tensor(tensor_name).dtype
    return None


def _prefix_counts(modules) -> Dict[str, int]:
    """Repartition des modules LoRA par branche (V6 : encodeur/decodeur)."""
    counts = {"decoder": 0, "encoder": 0, "self_conditioning": 0, "other": 0}
    for module in modules:
        if module.startswith("model.decoder.self_conditioning."):
            counts["self_conditioning"] += 1
        elif module.startswith("model.decoder."):
            counts["decoder"] += 1
        elif module.startswith("model.encoder."):
            counts["encoder"] += 1
        else:
            counts["other"] += 1
    return counts


def scale_adapter(adapter_dir, output_dir, factor: float,
                  dtype: str = "bfloat16", overwrite: bool = False,
                  log=print) -> dict:
    """Copie un adaptateur LoRA en multipliant `lora_B` par `factor`.

    Le facteur doit appartenir a `[0, 1]` : dans l'eq. (2), `lambda` est dans
    `[0,1]`, donc `1 - lambda` aussi. Un facteur nul est legitime (lambda=1 :
    transfert complet, l'adaptateur est neutralise sans etre fusionne).

    Retourne un manifeste (provenance, echelles, empreintes, comptages).
    """
    source = Path(adapter_dir)
    output = Path(output_dir)
    if not (0.0 <= factor <= 1.0):
        raise AdapterError(
            f"facteur d'echelle invalide : {factor} (attendu dans [0, 1])")
    target_dtype = _dtype_of(dtype)

    config_path = source / "adapter_config.json"
    if not config_path.exists():
        raise AdapterError(f"adapter_config.json introuvable dans {source}")
    with open(config_path) as f:
        config = json.load(f)
    r = int(config["r"])
    lora_alpha = float(config["lora_alpha"])
    use_rslora = bool(config.get("use_rslora"))
    scale_before = lora_alpha / (r ** 0.5) if use_rslora else lora_alpha / r
    scale_after = scale_before * factor

    weight_map, sharded = _adapter_layout(source)

    if output.exists():
        if output.is_file():
            raise AdapterError(f"sortie d'adaptateur invalide (fichier) : {output}")
        if any(output.iterdir()) and not overwrite:
            raise AdapterError(
                f"{output} existe deja et n'est pas vide. Relancer avec "
                "--overwrite pour remplacer l'adaptateur mis a l'echelle.")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=str(output.parent), prefix=output.name + ".tmp-"))

    try:
        for name in sorted(source.iterdir()):
            if name.name.endswith(".safetensors"):
                continue
            if name.name == "model.safetensors.index.json":
                continue
            if name.is_file():
                shutil.copy2(name, staging / name.name)

        by_file: Dict[str, list] = {}
        for tensor_name, fname in weight_map.items():
            by_file.setdefault(fname, []).append(tensor_name)

        n_scaled = 0
        n_copied = 0
        n_a_unchanged = 0
        for fname, tensor_names in sorted(by_file.items()):
            out_tensors = {}
            with safe_open(str(source / fname), framework="pt") as handle:
                metadata = handle.metadata()
                for tensor_name in sorted(tensor_names):
                    raw = handle.get_tensor(tensor_name)
                    if tensor_name.endswith(SCALE_SUFFIX):
                        scaled = (raw.to(torch.float32) * factor).to(target_dtype)
                        out_tensors[tensor_name] = scaled
                        n_scaled += 1
                    else:
                        out_tensors[tensor_name] = raw
                        n_copied += 1
                        if tensor_name.endswith(".lora_A.weight"):
                            n_a_unchanged += 1
            _atomic_save(out_tensors, staging / fname, metadata=metadata)
            log(f"    [adapter {fname}] {len(tensor_names)} tenseurs")

        if sharded:
            _atomic_write_json(
                staging / "model.safetensors.index.json",
                {"metadata": {"total_size": sum((staging / f).stat().st_size
                                                for f in by_file)},
                 "weight_map": weight_map})

        if output.exists():
            shutil.rmtree(output)
        os.replace(staging, output)
        staging = None
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)

    modules = {k[: -len(".lora_B.weight")] for k in weight_map
               if k.endswith(SCALE_SUFFIX)}
    manifest = {
        "unmerged": True,
        "source": str(source.resolve()),
        "output": str(output.resolve()),
        "factor": factor,
        "factor_applied_to": SCALE_SUFFIX.lstrip("."),
        "dtype": dtype,
        "r": r,
        "lora_alpha": lora_alpha,
        "lora_alpha_unchanged": True,
        "use_rslora": use_rslora,
        "scale_before": scale_before,
        "scale_after": scale_after,
        "n_modules": len(modules),
        "n_lora_B_scaled": n_scaled,
        "n_lora_A_unchanged": n_a_unchanged,
        "n_other_tensors_copied": n_copied - n_a_unchanged,
        "branches": _prefix_counts(modules),
        "sharded": sharded,
        "shards_sha256_before": shard_fingerprints(source, weight_map, limit_mb=64),
        "shards_sha256_after": shard_fingerprints(output, weight_map, limit_mb=64),
    }
    log(f"  Adaptateur mis a l'echelle x{factor} -> {output} "
        f"({n_scaled} lora_B, {len(modules)} modules)")
    return manifest


def verify_scaled_adapter(original_dir, scaled_dir, factor: float,
                          relative_tolerance: float | None = None,
                          log=print) -> dict:
    """Verification V5/V6 : `delta_scaled == factor * delta_original`.

    Recharge les deux adaptateurs par `LoraAdapterSource` (donc avec la meme
    formule PEFT que l'ecriture), compare les deltas FP32 module par module et
    controle que les deux branches encodeur/decodeur et les modules de
    self-conditioning sont preserves.

    `relative_tolerance` par defaut depend du dtype ecrit : 1e-2 en BF16
    (arrondi de `lora_B`) et 1e-3 en FP32.

    Cas `factor == 0` (lambda = 1) : le delta attendu est nul pour tous les
    modules ; la verification exige qu'il soit **exactement** nul et ne reclame
    pas de delta non nul (la direction n'est pas definie).
    """
    original = LoraAdapterSource(original_dir)
    scaled = LoraAdapterSource(scaled_dir)
    failures = []
    dtype = _b_weight_dtype(scaled_dir)
    if relative_tolerance is None:
        relative_tolerance = (MAX_RELATIVE_L2_ERROR_FP32 if dtype == torch.float32
                              else MAX_RELATIVE_L2_ERROR_BF16)
    try:
        if original.r != scaled.r:
            failures.append(f"r different : {original.r} -> {scaled.r}")
        if original.alpha != scaled.alpha:
            failures.append(
                f"lora_alpha modifie : {original.alpha} -> {scaled.alpha} "
                "(le facteur doit etre applique a lora_B, pas a lora_alpha)")
        modules_original = set(original.modules())
        modules_scaled = set(scaled.modules())
        missing = sorted(modules_original - modules_scaled)
        extra = sorted(modules_scaled - modules_original)
        if missing:
            failures.append(f"{len(missing)} module(s) absent(s) de l'adaptateur "
                            f"produit : {missing[:10]}")
        if extra:
            failures.append(f"{len(extra)} module(s) en trop dans l'adaptateur "
                            f"produit : {extra[:10]}")

        max_rel = 0.0
        max_rel_module = None
        min_cos = None
        min_cos_module = None
        n_compared = 0
        n_zero_delta = 0
        for module in sorted(modules_original & modules_scaled):
            delta = original.delta(module)
            scaled_delta = scaled.delta(module)
            expected = delta * factor
            expected_norm = float(torch.linalg.vector_norm(expected, dtype=torch.float64))
            diff_norm = float(torch.linalg.vector_norm(scaled_delta - expected,
                                                       dtype=torch.float64))
            n_compared += 1
            if expected_norm == 0.0:
                n_zero_delta += 1
                if diff_norm != 0.0:
                    if factor == 0:
                        failures.append(
                            f"{module}: facteur nul mais delta non nul "
                            f"(norme {diff_norm:.3e})")
                    else:
                        failures.append(f"{module}: delta attendu nul mais ecart "
                                        f"{diff_norm:.3e}")
                continue
            rel = diff_norm / expected_norm
            if rel > max_rel:
                max_rel, max_rel_module = rel, module
            cos = float((scaled_delta * expected).sum(dtype=torch.float64)
                        / (torch.linalg.vector_norm(scaled_delta, dtype=torch.float64)
                           * torch.linalg.vector_norm(expected, dtype=torch.float64)))
            if min_cos is None or cos < min_cos:
                min_cos, min_cos_module = cos, module

        if max_rel > relative_tolerance:
            failures.append(
                f"erreur L2 relative maximale {max_rel:.3e} > "
                f"{relative_tolerance:.0e} (module {max_rel_module})")
        if min_cos is not None and (1.0 - min_cos) > MAX_COSINE_DISTANCE:
            failures.append(
                f"cosinus minimal {min_cos:.9f} < {1.0 - MAX_COSINE_DISTANCE:.9f} "
                f"(module {min_cos_module})")
        # Un facteur nul neutralise l'adaptateur : le delta attendu est nul pour
        # TOUS les modules et la direction (cosinus) n'est pas definie. C'est un
        # cas legitime (lambda=1 : transfert complet) ; il exige seulement que
        # les deltas produits soient EXACTEMENT nuls (verifie ci-dessus).
        # L'erreur « aucun delta non nul » ne vaut que pour factor > 0 : elle
        # signale alors qu'aucune echelle n'a pu etre mesuree.
        cosine_skipped = n_compared == n_zero_delta
        if cosine_skipped and factor > 0:
            failures.append("aucun delta non nul : verification de l'echelle impossible")

        branches = _prefix_counts(modules_scaled)
        report = {
            "ok": not failures,
            "factor": factor,
            "zero_factor_exact": factor == 0 and not failures,
            "n_modules_original": len(modules_original),
            "n_modules_scaled": len(modules_scaled),
            "n_modules_compared": n_compared,
            "n_zero_delta_modules": n_zero_delta,
            "branches": branches,
            "max_relative_l2_error": max_rel,
            "max_relative_l2_module": max_rel_module,
            "min_cosine": min_cos,
            "min_cosine_module": min_cos_module,
            "criteria": {
                "max_relative_l2_error": relative_tolerance,
                "max_cosine_distance": MAX_COSINE_DISTANCE,
            },
            "dtype": str(dtype).replace("torch.", "") if dtype is not None else None,
            "failures": failures,
        }
    finally:
        original.close()
        scaled.close()

    log(f"  Verification adaptateur V5/V6 : "
        f"{'OK' if report['ok'] else 'ECHEC'} "
        f"(err L2 max {report['max_relative_l2_error']:.3e}, "
        f"cos min {report['min_cosine']})")
    return report
