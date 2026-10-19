#!/usr/bin/env python3
"""
Download Gemma 4, DiffusionGemma, Qwen2.5 and Dream checkpoints from HuggingFace.
Usage: python download_models.py [--output-dir ./checkpoints] [--set gemma|dream|all]

Jeux de modeles (identifiants HF verifies le 2026-10-19) :
  gemma : la famille DiffusionGemma du projet
  dream : la famille Qwen2.5-7B -> Dream-7B (A2D, Table 6 / Table 16)
          D_b = Dream-7B-Base (indispensable), D_p = Dream-7B-Instruct,
          A_b = Qwen2.5-7B, A_p = Qwen2.5-7B-Instruct
"""

import argparse
from huggingface_hub import snapshot_download

MODELS = {
    "gemma4_base": "google/gemma-4-26B-A4B",       # A_b (base AR)
    "gemma4_it":   "google/gemma-4-26B-A4B-it",    # A_p (post-trained AR)
    "diffusiongemma_it": "google/diffusiongemma-26B-A4B-it",  # D_p (diffusion post-trained)
}

# Famille A2D Qwen2.5-7B -> Dream-7B (A2D Tables 6 et 16).
DREAM_MODELS = {
    "qwen25_7b": "Qwen/Qwen2.5-7B",                 # A_b
    "qwen25_7b_it": "Qwen/Qwen2.5-7B-Instruct",     # A_p
    "dream_7b_base": "Dream-org/Dream-v0-Base-7B",  # D_b (indispensable)
    "dream_7b_instruct": "Dream-org/Dream-v0-Instruct-7B",  # D_p
}

SETS = {
    "gemma": MODELS,
    "dream": DREAM_MODELS,
    "all": {**MODELS, **DREAM_MODELS},
}

def main():
    parser = argparse.ArgumentParser(description="Download model checkpoints")
    parser.add_argument("--output-dir", default="checkpoints", help="Output directory")
    parser.add_argument("--set", choices=sorted(SETS), default="gemma",
                        help="Jeu de modeles a telecharger (defaut: gemma)")
    args = parser.parse_args()

    import os
    os.makedirs(args.output_dir, exist_ok=True)

    for name, repo_id in SETS[args.set].items():
        out = os.path.join(args.output_dir, name)
        if not os.path.exists(os.path.join(out, "model.safetensors.index.json")):
            print(f"Downloading {repo_id} -> {out}")
            snapshot_download(repo_id=repo_id, local_dir=out, max_workers=8)
        else:
            print(f"Already cached: {name}")

    print("All models downloaded.")

if __name__ == "__main__":
    main()
