#!/usr/bin/env python3
"""
Compute AR post-training task vector: tau_A = A_p - A_b.
Saves intermediate numpy files for later merge step.
Lecture via safetensors.safe_open (bits exacts), ~2-3 GB peak RAM.

Usage:
    python compute_tau_a.py \
        --base /path/to/gemma4_base \
        --posttrained /path/to/gemma4_it \
        --output-dir _artefact/tau_a
    # Economise ~35 Go : ne garde que le decoder mappable :
    python compute_tau_a.py --filter decoder ...
"""

import argparse
import datetime
import json
import os
import sys

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import gemma4_to_diffusiongemma, git_commit, tau_filename


def load_fp32(directory, fname, name):
    """Charge un tenseur en float32 numpy via safe_open (torch backend)."""
    with safe_open(os.path.join(directory, fname), framework="pt") as f:
        t = f.get_tensor(name)
    if isinstance(t, torch.Tensor):
        return t.to(torch.float32).numpy()
    return np.asarray(t, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Compute tau_A = A_p - A_b")
    parser.add_argument("--base", required=True, help="Path to AR base model (A_b)")
    parser.add_argument("--posttrained", required=True, help="Path to AR post-trained model (A_p)")
    parser.add_argument("--output-dir", default="_artefact/tau_a", help="Output directory for tau_A files")
    parser.add_argument("--filter", choices=["all", "decoder"], default="all",
                        help="'decoder' ne garde que les tenseurs mappables vers DiffusionGemma (economise ~35 Go)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Autorise la reutilisation d'un dossier de sortie non vide")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    existing = [f for f in os.listdir(args.output_dir) if f.endswith(".npy")]
    if existing and not args.overwrite:
        print(f"ERREUR: {args.output_dir} contient deja {len(existing)} .npy "
              f"(tau perime/incomplet possible). Relancer avec --overwrite "
              f"ou nettoyer le dossier.")
        raise SystemExit(2)

    for label, directory in (("A_b", args.base), ("A_p", args.posttrained)):
        if not os.path.exists(os.path.join(directory, "model.safetensors.index.json")):
            print(f"ERREUR: index introuvable pour {label}: {directory}")
            raise SystemExit(2)

    with open(os.path.join(args.base, "model.safetensors.index.json")) as f:
        map_b = json.load(f)["weight_map"]
    with open(os.path.join(args.posttrained, "model.safetensors.index.json")) as f:
        map_p = json.load(f)["weight_map"]

    common = sorted(set(map_b) & set(map_p))
    print(f"Common tensors (A_b <-> A_p): {len(common)}")
    if args.filter == "decoder":
        common = [n for n in common if gemma4_to_diffusiongemma(n) is not None]
        print(f"Filtre 'decoder': {len(common)} tenseurs mappables conserves")

    # Detection de collision de noms sanitizes (a.b -> a_b)
    seen = {}
    for n in common:
        fn = tau_filename(n)
        if fn in seen:
            print(f"ERREUR: collision de noms tau: '{seen[fn]}' et '{n}' "
                  f"-> meme fichier {fn}. Abandon (pas d'ecrasement silencieux).")
            raise SystemExit(1)
        seen[fn] = n

    import time
    t0 = time.time()
    for i, name in enumerate(common):
        if (i + 1) % 200 == 0 or i == len(common) - 1:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(common) - i - 1)
            print(f"  [{time.strftime('%H:%M:%S')}] {i+1}/{len(common)} "
                  f"({(i+1)/len(common)*100:.0f}%) — ETA ~{eta/60:.1f}min")

        tb = load_fp32(args.base, map_b[name], name)
        tp = load_fp32(args.posttrained, map_p[name], name)
        if tb.shape != tp.shape:
            print(f"ERREUR: shape mismatch {name}: A_b={tb.shape} vs A_p={tp.shape}. Abandon.")
            raise SystemExit(1)
        tau_a = tp - tb
        np.save(os.path.join(args.output_dir, tau_filename(name)), tau_a)

    manifest = {
        "base": os.path.abspath(args.base),
        "posttrained": os.path.abspath(args.posttrained),
        "filter": args.filter,
        "n_tensors": len(common),
        "date": datetime.datetime.now().astimezone().isoformat(),
        "git_commit": git_commit(),
    }
    with open(os.path.join(args.output_dir, "tau_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\ntau_A computed in {time.time()-t0:.1f}s. Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
