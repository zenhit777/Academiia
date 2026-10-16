#!/usr/bin/env python3
"""
Download Gemma 4 and DiffusionGemma checkpoints from HuggingFace.
Usage: python download_models.py [--output-dir ./checkpoints]
"""

import argparse
from huggingface_hub import snapshot_download

MODELS = {
    "gemma4_base": "google/gemma-4-26B-A4B",       # A_b (base AR)
    "gemma4_it":   "google/gemma-4-26B-A4B-it",    # A_p (post-trained AR)
    "diffusiongemma_it": "google/diffusiongemma-26B-A4B-it",  # D_p (diffusion post-trained)
}

def main():
    parser = argparse.ArgumentParser(description="Download model checkpoints")
    parser.add_argument("--output-dir", default="./checkpoints", help="Output directory")
    args = parser.parse_args()

    import os
    os.makedirs(args.output_dir, exist_ok=True)

    for name, repo_id in MODELS.items():
        out = os.path.join(args.output_dir, name)
        if not os.path.exists(os.path.join(out, "model.safetensors.index.json")):
            print(f"Downloading {repo_id} -> {out}")
            snapshot_download(repo_id=repo_id, local_dir=out, max_workers=8)
        else:
            print(f"Already cached: {name}")

    print("All models downloaded.")

if __name__ == "__main__":
    main()
