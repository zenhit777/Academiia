#!/usr/bin/env python3
"""
Mesure l'effet reel du merge Dp + alpha*tau_A APRES arrondi BF16.

Repond au point n°5 de l'audit : le delta theorique (0.8*tau) peut etre
partiellement absorbe par la quantification BF16. Ce script quantifie,
tenseur par tenseur, la proportion d'elements reellement modifies.

Methode (CPU seul, ~2 Go RAM) :
  pour chaque tenseur mappe Dp (via safe_open, bits BF16 exacts) :
    attendu_fp32 = Dp_fp32 + alpha * tau_A (npy float32)
    arrondi_BF16 = cast(attendu_fp32 -> bfloat16 -> float32)
    modifie = (bits(arrondi) != bits(Dp))

Usage:
    python scripts/measure_effect.py \
        --diffusiongemma checkpoints/diffusiongemma_it \
        --tau-dir _artefact/tau_a \
        --alphas 0.05,0.1,0.2,0.3,0.5,0.8 \
        --output _artefact/measure_effect.json
    # Variante "papier" (sans norms ni embeddings) :
    python scripts/measure_effect.py --exclude-norms-embed ...
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


def is_norm_or_embed(dg_name: str) -> bool:
    n = dg_name.lower()
    return (
        "layernorm" in n
        or "norm" in n
        or "embed_tokens" in n
        or n.endswith("layer_scalar")
    )


def g4_name_for(dg_name: str):
    if dg_name.startswith("model.decoder.layers."):
        return "model.language_model.layers." + dg_name[len("model.decoder.layers."):]
    if dg_name == "model.decoder.embed_tokens.weight":
        return "model.language_model.embed_tokens.weight"
    return None


def main():
    p = argparse.ArgumentParser(description="Mesure effet merge apres arrondi BF16")
    p.add_argument("--diffusiongemma", required=True)
    p.add_argument("--tau-dir", default="_artefact/tau_a")
    p.add_argument("--alphas", default="0.8",
                   help="Liste d'alphas separes par virgules (defaut: 0.8)")
    p.add_argument("--output", default="_artefact/measure_effect.json")
    p.add_argument("--limit", type=int, default=0,
                   help="Limite le nombre de tenseurs (0 = tous, pour test rapide)")
    p.add_argument("--exclude-norms-embed", action="store_true",
                   help="Exclut norms/embeddings (variante papier A2D)")
    args = p.parse_args()

    alphas = [float(a) for a in args.alphas.split(",") if a.strip()]
    dg_dir = Path(args.diffusiongemma)
    with open(dg_dir / "model.safetensors.index.json") as f:
        index = json.load(f)
    weight_map = index["weight_map"]

    # Regroupe par fichier pour ouvrir chaque shard une seule fois
    by_file = {}
    for name, fname in weight_map.items():
        by_file.setdefault(fname, []).append(name)

    per_alpha_changed = {a: 0 for a in alphas}
    per_alpha_total = {a: 0 for a in alphas}
    per_tensor = []
    n_mapped = 0
    n_skipped_variant = 0
    n_missing_tau = 0

    for fname in sorted(by_file.keys()):
        # Ouvre en torch/bf16 : bits exacts, pas de conversion maison
        with safe_open(dg_dir / fname, framework="pt") as f:
            names = sorted(by_file[fname])
            if args.limit and n_mapped >= args.limit:
                break
            for dg_name in names:
                if args.limit and n_mapped >= args.limit:
                    break
                g4 = g4_name_for(dg_name)
                if g4 is None:
                    continue
                if args.exclude_norms_embed and is_norm_or_embed(dg_name):
                    n_skipped_variant += 1
                    continue
                tau_path = os.path.join(args.tau_dir, g4.replace(".", "_") + ".npy")
                if not os.path.exists(tau_path):
                    n_missing_tau += 1
                    continue
                n_mapped += 1
                dp_bf16 = f.get_tensor(dg_name)  # torch.bfloat16
                dp_fp32 = dp_bf16.to(torch.float32).numpy()
                tau = np.load(tau_path)
                if tau.shape != dp_fp32.shape:
                    # Shape mismatch : le merge garde Dp inchange -> 0 modifie
                    rec = {"name": dg_name, "shape_mismatch": True,
                           "dp_shape": list(dp_fp32.shape),
                           "tau_shape": list(tau.shape)}
                    per_tensor.append(rec)
                    for a in alphas:
                        per_alpha_total[a] += dp_fp32.size
                    continue
                rec = {"name": dg_name, "size": int(dp_fp32.size),
                       "tau_l2": float(np.sqrt((tau ** 2).sum())),
                       "tau_maxabs": float(np.abs(tau).max()),
                       "by_alpha": {}}
                dp_bits = dp_bf16.view(torch.uint16).flatten()
                for a in alphas:
                    expected = dp_fp32 + a * tau
                    rounded = torch.from_numpy(
                        np.ascontiguousarray(expected)).to(torch.bfloat16)
                    changed = int((rounded.view(torch.uint16).flatten() != dp_bits).sum())
                    total = dp_fp32.size
                    per_alpha_changed[a] += changed
                    per_alpha_total[a] += total
                    applied = (rounded.to(torch.float32).numpy() - dp_fp32)
                    rec["by_alpha"][str(a)] = {
                        "frac_changed": changed / total,
                        "applied_l2": float(np.sqrt((applied ** 2).sum())),
                        "applied_maxabs": float(np.abs(applied).max()),
                    }
                per_tensor.append(rec)

    summary = {
        "diffusiongemma": str(args.diffusiongemma),
        "tau_dir": str(args.tau_dir),
        "exclude_norms_embed": args.exclude_norms_embed,
        "tensors_mapped": n_mapped,
        "tensors_skipped_variant": n_skipped_variant,
        "tensors_missing_tau": n_missing_tau,
        "by_alpha_global": {
            str(a): {
                "frac_changed": (per_alpha_changed[a] / per_alpha_total[a]
                                 if per_alpha_total[a] else 0.0),
                "n_changed": per_alpha_changed[a],
                "n_total": per_alpha_total[a],
            } for a in alphas
        },
    }
    out = {"summary": summary, "per_tensor": per_tensor}
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Tenseurs mappes mesures : {n_mapped} "
          f"(variante exclue: {n_skipped_variant}, tau manquant: {n_missing_tau})")
    for a in alphas:
        g = summary["by_alpha_global"][str(a)]
        print(f"  alpha={a:<5} modifies: {g['frac_changed']*100:5.2f}% "
              f"({g['n_changed']}/{g['n_total']} elements)")
    print(f"Detail JSON : {args.output}")


if __name__ == "__main__":
    main()
