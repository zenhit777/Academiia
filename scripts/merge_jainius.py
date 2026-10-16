#!/usr/bin/env python3
"""
Apply merge: Jainius = D_p + alpha * tau_A_mapped.
Ablation « Dp + ατA » inspiree A2D (PAS le A2D-Merge complet : pas de τ_D).
Maps Gemma 4 tensor names to DiffusionGemma decoder layer names.
Lecture Dp via safetensors.safe_open (bits exacts).
Memory-efficient: writes output files incrementally (~5GB max in RAM).

Usage:
    python merge_jainius.py \
        --diffusiongemma /path/to/diffusiongemma_it \
        --tau-dir _artefact/tau_a \
        --output /path/to/Jainius \
        [--alpha 0.3] [--exclude-norms-embed]
"""

import argparse
import datetime
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (diffusiongemma_to_gemma4, git_commit, is_norm_or_embed,
                     list_source_sidecar_files, tau_filename)


def main():
    parser = argparse.ArgumentParser(description="Merge: Jainius = D_p + alpha * tau_A")
    parser.add_argument("--diffusiongemma", required=True, help="Path to DiffusionGemma IT checkpoint (D_p)")
    parser.add_argument("--tau-dir", default="_artefact/tau_a", help="Directory with tau_A numpy files")
    parser.add_argument("--output", required=True, help="Output directory for Jainius checkpoint")
    parser.add_argument("--alpha", type=float, default=0.3, help="Interpolation weight (default: 0.3, 0.8 deprecie)")
    parser.add_argument("--exclude-norms-embed", action="store_true",
                        help="Exclut embeddings/normalisations (variante papier A2D)")
    parser.add_argument("--on-shape-mismatch", choices=["error", "warn"], default="error",
                        help="Comportement si tau_A et D_p ont des shapes differentes (defaut: error)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Autorise la reutilisation d'un dossier de sortie existant (nettoie les shards obsoletes)")
    args = parser.parse_args()

    if not 0 <= args.alpha <= 1:
        print(f"ERREUR: --alpha={args.alpha} hors [0,1].")
        raise SystemExit(2)

    os.makedirs(args.output, exist_ok=True)
    pre_existing = [f for f in os.listdir(args.output) if f.endswith(".safetensors")]
    if pre_existing and not args.overwrite:
        print(f"ERREUR: {args.output} contient deja {len(pre_existing)} shards. "
              f"Relancer avec --overwrite (nettoyage des shards obsoletes) "
              f"ou vers un dossier vide. Pas de merge partiel.")
        raise SystemExit(2)

    index_path = os.path.join(args.diffusiongemma, "model.safetensors.index.json")
    with open(index_path) as f:
        dg_index = json.load(f)
    dg_weight_map = dg_index["weight_map"]

    # Exige le manifest tau (traçabilite : pas de tau perime/incomplet silencieux)
    tau_manifest_path = os.path.join(args.tau_dir, "tau_manifest.json")
    tau_manifest = None
    if os.path.exists(tau_manifest_path):
        with open(tau_manifest_path) as f:
            tau_manifest = json.load(f)
    else:
        print(f"ATTENTION: {tau_manifest_path} absent (tau calcule avec "
              f"l'ancien script ?). Mapping base sur l'existence des .npy, "
              f"sans controle de provenance.")

    # Build mapping dg_name -> g4_name (+ detection collision fichier tau)
    mapping = {}
    seen_files = {}
    for dg_name in dg_weight_map:
        g4_name = diffusiongemma_to_gemma4(dg_name)
        if g4_name is None:
            continue
        if args.exclude_norms_embed and is_norm_or_embed(dg_name):
            continue
        tau_path = os.path.join(args.tau_dir, tau_filename(g4_name))
        if os.path.exists(tau_path):
            if tau_path in seen_files:
                print(f"ERREUR: collision tau: {tau_path} pour "
                      f"'{seen_files[tau_path]}' et '{dg_name}'. Abandon.")
                raise SystemExit(1)
            seen_files[tau_path] = dg_name
            mapping[dg_name] = g4_name

    print(f"Mapped tensors (Gemma4 -> DiffusionGemma): {len(mapping)}"
          + (" (variante sans norms/embed)" if args.exclude_norms_embed else ""))

    dg_by_file = {}
    for dg_name, fname in dg_weight_map.items():
        dg_by_file.setdefault(fname, []).append(dg_name)

    print("\nWriting output files (incremental, ~5GB/file)...")
    import time
    t0 = time.time()
    total_written = 0
    total_tensors = len(dg_weight_map)
    shape_errors = []

    for n_file, fname in enumerate(sorted(dg_by_file.keys()), 1):
        tensors_in_file = sorted(dg_by_file[fname])
        tensor_dict = {}
        with safe_open(os.path.join(args.diffusiongemma, fname), framework="pt") as f:
            for dg_name in tensors_in_file:
                d_p = f.get_tensor(dg_name).to(torch.float32).numpy()
                if dg_name in mapping:
                    g4_name = mapping[dg_name]
                    tau_a = np.load(os.path.join(args.tau_dir, tau_filename(g4_name)))
                    if tau_a.shape != d_p.shape:
                        msg = (f"{dg_name}: tau_A={tau_a.shape} vs "
                               f"D_p={d_p.shape}")
                        if args.on_shape_mismatch == "error":
                            print(f"  SHAPE ERROR: {msg} — abandon (--on-shape-mismatch=error)")
                            raise SystemExit(1)
                        print(f"  SHAPE WARNING: {msg} — D_p inchange")
                        shape_errors.append(msg)
                        jainius = d_p
                    else:
                        jainius = d_p + args.alpha * tau_a
                else:
                    jainius = d_p
                tensor_dict[dg_name] = torch.from_numpy(
                    np.ascontiguousarray(jainius)).to(torch.bfloat16)

        # Ecriture atomique : tmp dans le dossier cible + replace
        fd, tmp_path = tempfile.mkstemp(dir=args.output, prefix=fname + ".tmp-")
        os.close(fd)
        try:
            save_file(tensor_dict, tmp_path)
            os.replace(tmp_path, os.path.join(args.output, fname))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        del tensor_dict
        total_written += len(tensors_in_file)
        elapsed = time.time() - t0
        done = total_written / total_tensors
        eta = elapsed / done * (1 - done) if done else 0
        print(f"    [{time.strftime('%H:%M:%S')}] fichier {n_file}/{len(dg_by_file)}: "
              f"{fname} ({len(tensors_in_file)} tenseurs) — "
              f"total {total_written}/{total_tensors} — ETA ~{eta/60:.1f}min")

    # Nettoie les shards obsoletes d'une execution precedente
    expected_files = set(dg_by_file.keys())
    for entry in os.listdir(args.output):
        if entry.endswith(".safetensors") and entry not in expected_files:
            print(f"  Nettoyage shard obsolete : {entry}")
            os.remove(os.path.join(args.output, entry))

    # Index + total_size (standard HF)
    weight_map = {dg_name: fname for fname, names in dg_by_file.items() for dg_name in names}
    total_size = 0
    for fname in expected_files:
        total_size += os.path.getsize(os.path.join(args.output, fname))
    index_out = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    with open(os.path.join(args.output, "model.safetensors.index.json"), "w") as f:
        json.dump(index_out, f, indent=2)

    # Recopie TOUS les sidecars source (pas de liste en dur : LICENSE, etc. preservés)
    for entry in list_source_sidecar_files(args.diffusiongemma):
        src = os.path.join(args.diffusiongemma, entry)
        dst = os.path.join(args.output, entry)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)

    manifest = {
        "method": "Dp + alpha*tau_A (ablation inspiree A2D, pas de tau_D)",
        "alpha": args.alpha,
        "exclude_norms_embed": args.exclude_norms_embed,
        "diffusiongemma": os.path.abspath(args.diffusiongemma),
        "tau_dir": os.path.abspath(args.tau_dir),
        "tau_manifest": tau_manifest,
        "n_mapped": len(mapping),
        "n_unchanged": len(weight_map) - len(mapping),
        "n_tensors": len(weight_map),
        "shape_errors": shape_errors,
        "date": datetime.datetime.now().astimezone().isoformat(),
        "git_commit": git_commit(),
    }
    with open(os.path.join(args.output, "merge_config.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nJainius checkpoint saved to: {args.output}")
    print(f"  Total tensors: {len(weight_map)}")
    print(f"  Mapped (merge applique): {len(mapping)}")
    print(f"  Unchanged (DiffusionGemma-only): {len(weight_map) - len(mapping)}")
    print(f"  Manifest: merge_config.json")
    print(f"  Time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
