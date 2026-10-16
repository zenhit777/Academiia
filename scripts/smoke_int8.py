#!/usr/bin/env python3
"""
Test rapide INT8 (transformers + bitsandbytes) — itération, PAS preuve finale.

Charge un checkpoint DiffusionGemma/Jainius (~52 Go BF16 -> ~27 Go INT8,
tient en 31 Go VRAM batch 1) et verifie :
  1. chargement sans OOM (device_map auto + offload disque si besoin),
  2. forward passe : logits finis sur un prompt court,
  3. generate() si l'API diffusion du transformers installe le supporte.

La preuve publiee reste le chemin BF16 + vLLM offloade (fidélite max) :
le bruit INT8 est du meme ordre que le delta 0.8*tau (cf. measure_effect.py).

Usage :
    python scripts/smoke_int8.py --checkpoint checkpoints/diffusiongemma_it
    python scripts/smoke_int8.py --checkpoint checkpoints/Jainius --no-generate
"""

import argparse
import os

import torch


def main():
    p = argparse.ArgumentParser(description="Smoke test INT8 rapide (non-preuve)")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--prompt", default="En une phrase, qu'est-ce que la photosynthese ?")
    p.add_argument("--no-generate", action="store_true", help="Forward seul, sans generate()")
    p.add_argument("--max-memory-gpu", default="28GiB")
    p.add_argument("--max-memory-cpu", default="60GiB")
    p.add_argument("--offload-folder", default="_artefact/offload")
    args = p.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.makedirs(args.offload_folder, exist_ok=True)
    print(f"Chargement {args.checkpoint} (INT8, device_map auto)...")
    tok = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        trust_remote_code=False,
        load_in_8bit=True,
        device_map="auto",
        max_memory={"cuda:0": args.max_memory_gpu, "cpu": args.max_memory_cpu},
        offload_folder=args.offload_folder,
        # Protege routage MoE + normes (cf. variante papier) :
        llm_int8_skip_modules=["router", "norm", "layernorm", "embed", "lm_head"],
        llm_int8_threshold=6.0,
    )
    model.eval()
    print("  chargement OK")

    inputs = tok(args.prompt, return_tensors="pt")
    # Envoie les input_ids sur le device du premier module (accelerate disperse)
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model(**inputs)
    logits = out.logits.float()
    finite = torch.isfinite(logits).all().item()
    print(f"  forward OK : logits shape {tuple(logits.shape)}, finis={finite}")
    if not finite:
        print("SMOKE FAIL : NaN/Inf en forward.")
        raise SystemExit(1)

    if args.no_generate:
        print("SMOKE OK (forward seul).")
        return

    print("Tentative generate() ...")
    try:
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        text = tok.decode(gen[0], skip_special_tokens=True)
        print(f"  generation OK : {text[:500]!r}")
        print("SMOKE OK.")
    except Exception as e:
        print(f"  generate() non supporte via transformers ({type(e).__name__}: {e}).")
        print("  Forward OK mais echantillonnage diffusion -> passer par vLLM (BF16 offloade).")
        print("SMOKE PARTIEL (forward seul).")


if __name__ == "__main__":
    main()
