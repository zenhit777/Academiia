#!/usr/bin/env python3
"""
Mesure l'effet reel du merge Dp + alpha*tau_A APRES arrondi BF16.

Pourquoi ce script : le « % d'elements modifies » NE mesure PAS l'ampleur du
changement. Un poids proche de zero a un ULP BF16 minuscule et « bouge »
(valeur binaire differente) pour un delta numeriquement insignifiant ; a
l'inverse un poids de grand module peut rester « inchange » malgre un delta non
negligeable. Ce script rapporte donc, en plus du % modifie, des indicateurs
d'AMPLEUR :

  - dp_l2          : ||Dp||_2 (module du tenseur d'origine)
  - ratio_rel      : ||alpha*tau|| / ||Dp|| (taille du delta relativement a Dp)
  - fidelite_norme : ||applied|| / ||alpha*tau|| (part du delta theorique qui
                     survit a l'arrondi BF16 ; 1.0 = arrondi negligeable)
  - cosinus        : cos(applied, alpha*tau) (direction preservee)

Methode (CPU seul) :
  pour chaque tenseur mappe Dp (safe_open, bits BF16 exacts) et chaque alpha :
    attendu = Dp_fp32 + alpha * tau_A (npy float32)
    applied = cast(attendu -> bfloat16 -> float32) - Dp_fp32
    modifie = (bits(applied) != 0)

Usage :
    python modules/a2d/legacy/scripts/measure_effect.py \
        --diffusiongemma checkpoints/diffusiongemma_it \
        --tau-dir _artefact/tau_a \
        --alphas 0.05,0.1,0.2,0.3 \
        --output _artefact/measure_effect_v2.json
    # Variante NON conforme a l'App. A.2 : INCLURE norms/embeddings (exclus par
    # defaut). Pour DiffusionGemma, l'App. A.2 dit que les embeddings, les normes
    # et la tete de sortie restent INCHANGES ; cette variante n'est donc pas la
    # configuration du papier (rectificatif J6bis : MIN-06).
    python modules/a2d/legacy/scripts/measure_effect.py --include-norms-embed ...
"""

import argparse
import json
import os

import numpy as np
import torch
from safetensors import safe_open

from academiia.a2d.mapping import (diffusiongemma_to_gemma4 as g4_name_for, family_of,
                                    is_norm_or_embed, tau_filename)

FAMILIES = ["attn", "mlp/experts", "norms", "embed", "router", "autre"]


def _pct(values, q):
    return float(np.percentile(values, q)) if values else None


def main():
    p = argparse.ArgumentParser(description="Mesure effet merge apres arrondi BF16")
    p.add_argument("--diffusiongemma", required=True)
    p.add_argument("--tau-dir", default="_artefact/tau_a")
    p.add_argument("--alphas", default="0.05,0.1,0.2,0.3",
                   help="Liste d'alphas separes par virgules (defaut: 0.05,0.1,0.2,0.3)")
    p.add_argument("--output", default="_artefact/measure_effect_v2.json")
    p.add_argument("--limit", type=int, default=0,
                   help="Limite le nombre de tenseurs (0 = tous, pour test rapide)")
    p.add_argument("--include-norms-embed", action="store_true",
                   help="INCLUT norms/embeddings (defaut : exclus, variante papier A2D)")
    args = p.parse_args()

    alphas = [float(a) for a in args.alphas.split(",") if a.strip()]
    dg_dir = args.diffusiongemma
    with open(os.path.join(dg_dir, "model.safetensors.index.json")) as f:
        index = json.load(f)
    weight_map = index["weight_map"]

    # Regroupe par fichier pour ouvrir chaque shard une seule fois
    by_file = {}
    for name, fname in weight_map.items():
        by_file.setdefault(fname, []).append(name)

    per_alpha_changed = {a: 0 for a in alphas}
    per_alpha_total = {a: 0 for a in alphas}
    per_tensor = []
    # Agregation par famille puis par alpha
    fam = {fam_name: {a: {"ratio_rel": [], "cos_num": 0.0, "cos_den": 0,
                          "fidelite": [], "n_changed": 0, "n_elements": 0,
                          "n_tensors": 0}
                      for a in alphas}
           for fam_name in FAMILIES}
    n_mapped = 0
    n_skipped_variant = 0
    n_missing_tau = 0

    for fname in sorted(by_file.keys()):
        # Ouvre en torch/bf16 : bits exacts, pas de conversion maison
        with safe_open(os.path.join(dg_dir, fname), framework="pt") as f:
            for dg_name in sorted(by_file[fname]):
                if args.limit and n_mapped >= args.limit:
                    break
                g4 = g4_name_for(dg_name)
                if g4 is None:
                    continue
                if not args.include_norms_embed and is_norm_or_embed(dg_name):
                    n_skipped_variant += 1
                    continue
                tau_path = os.path.join(args.tau_dir, tau_filename(g4))
                if not os.path.exists(tau_path):
                    n_missing_tau += 1
                    continue
                n_mapped += 1
                family = family_of(dg_name)
                dp_bf16 = f.get_tensor(dg_name)  # torch.bfloat16
                dp_fp32 = dp_bf16.to(torch.float32).numpy()
                tau = np.load(tau_path)
                if tau.shape != dp_fp32.shape:
                    # Shape mismatch : le merge garde Dp inchange -> 0 modifie
                    per_tensor.append({"name": dg_name, "family": family,
                                       "shape_mismatch": True,
                                       "dp_shape": list(dp_fp32.shape),
                                       "tau_shape": list(tau.shape)})
                    for a in alphas:
                        per_alpha_total[a] += dp_fp32.size
                        fam[family][a]["n_elements"] += dp_fp32.size
                        fam[family][a]["n_tensors"] += 1
                    continue

                dp_l2 = float(np.sqrt((dp_fp32.astype(np.float64) ** 2).sum()))
                tau_l2 = float(np.sqrt((tau.astype(np.float64) ** 2).sum()))
                rec = {"name": dg_name, "family": family, "size": int(dp_fp32.size),
                       "dp_l2": dp_l2,
                       "tau_l2": tau_l2,
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
                    applied64 = applied.astype(np.float64)
                    applied_l2 = float(np.sqrt((applied64 ** 2).sum()))
                    alpha_tau_l2 = a * tau_l2
                    ratio_rel = (alpha_tau_l2 / dp_l2) if dp_l2 > 0 else None
                    fidelite = (applied_l2 / alpha_tau_l2) if alpha_tau_l2 > 0 else None
                    if applied_l2 > 0 and tau_l2 > 0:
                        cos = float((applied64 * tau.astype(np.float64)).sum()
                                    / (applied_l2 * tau_l2))
                    else:
                        cos = None
                    rec["by_alpha"][str(a)] = {
                        "frac_changed": changed / total,
                        "applied_l2": applied_l2,
                        "applied_maxabs": float(np.abs(applied).max()),
                        "ratio_rel": ratio_rel,
                        "fidelite_norme": fidelite,
                        "cosinus": cos,
                    }
                    fa = fam[family][a]
                    fa["n_tensors"] += 1
                    fa["n_elements"] += total
                    fa["n_changed"] += changed
                    if ratio_rel is not None:
                        fa["ratio_rel"].append(ratio_rel)
                    if fidelite is not None:
                        fa["fidelite"].append(fidelite)
                    if cos is not None:
                        fa["cos_num"] += total * cos
                        fa["cos_den"] += total
                per_tensor.append(rec)

    by_family = {}
    for fam_name in FAMILIES:
        entry = {}
        for a in alphas:
            fa = fam[fam_name][a]
            entry[str(a)] = {
                "n_tensors": fa["n_tensors"],
                "n_elements": fa["n_elements"],
                "frac_changed": (fa["n_changed"] / fa["n_elements"]
                                 if fa["n_elements"] else None),
                "ratio_rel_median": _pct(fa["ratio_rel"], 50),
                "ratio_rel_p95": _pct(fa["ratio_rel"], 95),
                "cosinus_size_weighted": (fa["cos_num"] / fa["cos_den"]
                                          if fa["cos_den"] else None),
                "fidelite_median": _pct(fa["fidelite"], 50),
            }
        if entry[str(alphas[0])]["n_tensors"]:
            by_family[fam_name] = entry

    summary = {
        "diffusiongemma": str(args.diffusiongemma),
        "tau_dir": str(args.tau_dir),
        "include_norms_embed": args.include_norms_embed,
        "alphas": alphas,
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
        "by_family": by_family,
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
    for fam_name, entry in by_family.items():
        a0 = str(alphas[0])
        e = entry[a0]
        print(f"  [{fam_name:11s}] ratio_rel med={e['ratio_rel_median']} "
              f"p95={e['ratio_rel_p95']} cos={e['cosinus_size_weighted']}")
    print(f"Detail JSON : {args.output}")


if __name__ == "__main__":
    main()
