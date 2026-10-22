#!/usr/bin/env python3
"""Geometrie des vecteurs de post-training (A2D Table 21 / App. B.7).

Calcule, sur les tenseurs du *backbone* (attention, MLP, normalisations ; hors
embeddings et hors tete de sortie, conformement a la caption de la Table 21 et
a App. B.7 du papier) :

    ||tau_A||, ||tau_D||, ||Delta_b||, ||Delta_p||,
    cos(Delta_b, Delta_p), cos(tau_A, tau_D)

avec, d'apres App. B.7 :

    tau_A = A_p - A_b        tau_D = D_p - D_b
    Delta_b = D_b - A_b      Delta_p = D_p - A_p

Lecture seule (safetensors.safe_open), rien n'est ecrit sur disque.

Exemple (famille Qwen2.5-7B -> Dream-7B du papier) :

    python scripts/measure_geometry.py \
        --ar-base checkpoints/qwen25_7b --ar-post checkpoints/qwen25_7b_it \
        --diffusion-base checkpoints/dream_7b_base \
        --diffusion-post checkpoints/dream_7b_instruct \
        --output _artefact/j6/geometry_dream.json
"""

import argparse
import json
import math
import os
from collections import defaultdict

import torch

from academiia.a2d import CheckpointPairSource, family_of, index_fingerprint, module_of

DEFAULT_FAMILIES = ("attn", "mlp", "norms")


def norm2(delta: torch.Tensor) -> float:
    return float(delta.double().pow(2).sum().item())


def dot(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.double() * b.double()).sum().item())


def main() -> int:
    parser = argparse.ArgumentParser(description="Geometrie des vecteurs (A2D Table 21)")
    parser.add_argument("--ar-base", required=True, help="A_b (AR base)")
    parser.add_argument("--ar-post", required=True, help="A_p (AR post-trained)")
    parser.add_argument("--diffusion-base", required=True, help="D_b (diffusion base)")
    parser.add_argument("--diffusion-post", required=True, help="D_p (diffusion post-trained)")
    parser.add_argument("--families", default=",".join(DEFAULT_FAMILIES),
                        help="Familles de tenseurs incluses (defaut: backbone A2D Table 21)")
    parser.add_argument("--output", default=None, help="Fichier JSON de sortie")
    args = parser.parse_args()

    families = {f.strip() for f in args.families.split(",") if f.strip()}

    ar_pair = CheckpointPairSource(args.ar_post, args.ar_base)
    diff_pair = CheckpointPairSource(args.diffusion_post, args.diffusion_base)

    names = set(ar_pair.modules()) & set(diff_pair.modules())
    print(f"Tenseurs communs AR/diffusion : {len(names)}")

    acc = defaultdict(lambda: defaultdict(float))
    counts = defaultdict(int)
    skipped = defaultdict(int)
    for name in sorted(names):
        family = family_of(name)
        if family not in families:
            skipped[family] += 1
            continue
        module = module_of(name)
        # tau_A = A_p - A_b ; tau_D = D_p - D_b ; Delta_b = D_b - A_b ; Delta_p = D_p - A_p
        a_p = ar_pair.tensor(module, "post")
        a_b = ar_pair.tensor(module, "base")
        d_p = diff_pair.tensor(module, "post")
        d_b = diff_pair.tensor(module, "base")
        tau_a = a_p - a_b
        tau_d = d_p - d_b
        delta_b = d_b - a_b
        delta_p = d_p - a_p
        st = acc[family]
        st["tau_a"] += norm2(tau_a)
        st["tau_d"] += norm2(tau_d)
        st["delta_b"] += norm2(delta_b)
        st["delta_p"] += norm2(delta_p)
        st["cos_bp_num"] += dot(delta_b, delta_p)
        st["cos_ad_num"] += dot(tau_a, tau_d)
        counts[family] += 1
        if counts[family] % 50 == 0:
            print(f"  {family}: {counts[family]} tenseurs")

    def cos(num, x2, y2):
        if x2 <= 0 or y2 <= 0:
            return None
        return num / math.sqrt(x2 * y2)

    families_out = {}
    totals = defaultdict(float)
    for family, st in sorted(acc.items()):
        entry = {
            "n_tensors": counts[family],
            "norm_tau_A": math.sqrt(st["tau_a"]),
            "norm_tau_D": math.sqrt(st["tau_d"]),
            "norm_Delta_b": math.sqrt(st["delta_b"]),
            "norm_Delta_p": math.sqrt(st["delta_p"]),
            "cos_Delta_b_Delta_p": cos(st["cos_bp_num"], st["delta_b"], st["delta_p"]),
            "cos_tau_A_tau_D": cos(st["cos_ad_num"], st["tau_a"], st["tau_d"]),
        }
        families_out[family] = entry
        for key in ("tau_a", "tau_d", "delta_b", "delta_p", "cos_bp_num", "cos_ad_num"):
            totals[key] += st[key]
    global_entry = {
        "n_tensors": sum(counts.values()),
        "norm_tau_A": math.sqrt(totals["tau_a"]),
        "norm_tau_D": math.sqrt(totals["tau_d"]),
        "norm_Delta_b": math.sqrt(totals["delta_b"]),
        "norm_Delta_p": math.sqrt(totals["delta_p"]),
        "cos_Delta_b_Delta_p": cos(totals["cos_bp_num"], totals["delta_b"], totals["delta_p"]),
        "cos_tau_A_tau_D": cos(totals["cos_ad_num"], totals["tau_a"], totals["tau_d"]),
    }

    report = {
        "sources": {
            "ar_base": {"path": os.path.abspath(args.ar_base),
                        "index": index_fingerprint(args.ar_base)},
            "ar_post": {"path": os.path.abspath(args.ar_post),
                        "index": index_fingerprint(args.ar_post)},
            "diffusion_base": {"path": os.path.abspath(args.diffusion_base),
                                "index": index_fingerprint(args.diffusion_base)},
            "diffusion_post": {"path": os.path.abspath(args.diffusion_post),
                                "index": index_fingerprint(args.diffusion_post)},
        },
        "families_included": sorted(families),
        "per_family": families_out,
        "global": global_entry,
        "skipped_tensors_per_family": dict(skipped),
        "reference_targets_a2d_table21": {
            "Qwen2.5-7B -> Dream-7B": {"norm_tau_A": 23.0, "norm_tau_D": 31.5,
                                        "norm_Delta_b": 729, "norm_Delta_p": 732,
                                        "cos_Delta_b_Delta_p": 0.999,
                                        "cos_tau_A_tau_D": 0.013},
        },
    }
    print(json.dumps({"global": global_entry, "per_family": families_out}, indent=2))
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Rapport ecrit : {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
