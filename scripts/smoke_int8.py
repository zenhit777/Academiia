#!/usr/bin/env python3
"""
 Test rapide INT8 (transformers + bitsandbytes) — itération, PAS preuve finale.

 ATTENTION : ce script est un test de NON-CASSE UNIQUEMENT. Il ne peut pas départager
 Dp et Jainius ni mesurer la qualité d'inférence. Le forward passe sur un prompt court,
 mais la génération n'est pas évaluée quantitativement.

Usage :
    python scripts/smoke_int8.py --checkpoint checkpoints/diffusiongemma_it
    python scripts/smoke_int8.py --checkpoint checkpoints/Jainius --no-generate
"""
 # Prérequis : pip install pillow (AutoProcessor/Gemma4Processor en dépend).

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

    # DiffusionGemmaForBlockDiffusion est la classe officielle pour cette architecture.
    # Fallback AutoModelForCausalLM si transformers < 5.11 (architecture non enregistree).
    try:
        from transformers import DiffusionGemmaForBlockDiffusion, AutoProcessor
        use_diffusiongemma = True
    except ImportError:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        use_diffusiongemma = False
    os.makedirs(args.offload_folder, exist_ok=True)
    print(f"Chargement {args.checkpoint} ...")

    if use_diffusiongemma:
        # Chargement INT8 via DiffusionGemmaForBlockDiffusion (classe officielle)
        try:
            processor = AutoProcessor.from_pretrained(args.checkpoint, trust_remote_code=False)
            model = DiffusionGemmaForBlockDiffusion.from_pretrained(
                args.checkpoint,
                load_in_8bit=True,
                device_map="auto",
                max_memory={0: args.max_memory_gpu, "cpu": args.max_memory_cpu},
                offload_folder=args.offload_folder,
                llm_int8_skip_modules=["router", "norm", "layernorm", "embed"],
                llm_int8_threshold=6.0,
            )
            mode = "INT8"
        except Exception as e:
            print(f"  INT8 indisponible ({type(e).__name__}: {e})")
            print("  Fallback BF16 + device_map auto ...")
            model = DiffusionGemmaForBlockDiffusion.from_pretrained(
                args.checkpoint,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                max_memory={0: args.max_memory_gpu, "cpu": args.max_memory_cpu},
                offload_folder=args.offload_folder,
            )
            mode = "BF16"
    else:
        # Fallback pour transformers < 5.11 (architecture non enregistree)
        try:
            model = AutoModelForCausalLM.from_pretrained(
                args.checkpoint,
                trust_remote_code=False,
                load_in_8bit=True,
                device_map="auto",
                max_memory={0: args.max_memory_gpu, "cpu": args.max_memory_cpu},
                offload_folder=args.offload_folder,
                llm_int8_skip_modules=["router", "norm", "layernorm", "embed", "lm_head"],
                llm_int8_threshold=6.0,
            )
            mode = "INT8"
        except Exception as e:
            print(f"  INT8 indisponible ({type(e).__name__}: {e})")
            print("  Fallback BF16 + device_map auto ...")
            model = AutoModelForCausalLM.from_pretrained(
                args.checkpoint,
                trust_remote_code=False,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                max_memory={0: args.max_memory_gpu, "cpu": args.max_memory_cpu},
                offload_folder=args.offload_folder,
            )
            mode = "BF16"
    model.eval()
    print(f"  chargement OK ({mode})")
    # Preparation du prompt : AutoProcessor avec chat template (DiffusionGemma),
    # fallback AutoTokenizer si pas de processor.
    if use_diffusiongemma and "processor" in locals():
        messages = [{"role": "user", "content": args.prompt}]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        )
    else:
        tok = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=False)
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
    # Mémoire GPU allouée (utile pour estimer la VRAM nécessaire)
    try:
        mem = torch.cuda.max_memory_allocated() / 2**30
        print(f"  max_memory_allocated: {mem:.1f} Go")
    except Exception:
        pass
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
        text = tok.decode(gen[0], skip_special_tokens=True) if "tok" in locals() else processor.tokenizer.decode(gen[0], skip_special_tokens=True)
        print(f"  generation OK : {text[:500]!r}")
        print("SMOKE OK.")
    except Exception as e:
        print(f"  generate() non supporte via transformers ({type(e).__name__}: {e}).")
        print("  Forward OK mais echantillonnage diffusion -> passer par vLLM (BF16 offloade).")
        print("SMOKE PARTIEL (forward seul).")


if __name__ == "__main__":
    main()
