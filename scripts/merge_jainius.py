#!/usr/bin/env python3
"""
Apply A2D merge: Jainius = D_p + alpha * tau_A_mapped.
Maps Gemma 4 tensor names to DiffusionGemma decoder layer names.
Memory-efficient: writes output files incrementally (~5GB max in RAM).

Usage:
    python merge_jainius.py \
        --diffusiongemma /path/to/diffusiongemma_it \
        --tau-dir _artefact/tau_a \
        --output /path/to/Jainius \
        [--alpha 0.8]
"""

import argparse
import json
import os
import shutil
import struct
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import save_file


def bf16_bits_to_fp32(arr_u16):
    """Convert BF16 bit-pattern (uint16) to float32 values (voir compute_tau_a.py)."""
    u32 = arr_u16.astype(np.uint32) << np.uint32(16)
    return u32.view(np.float32)


class SafetensorsMmapLoader:
    """Memory-mapped loader for multi-file safetensors files."""

    def __init__(self, directory):
        self.directory = Path(directory)
        index_path = self.directory / "model.safetensors.index.json"
        with open(index_path) as f:
            self.index = json.load(f)

        self.tensor_info = {}  # name -> (fname, kind, data_start, offset, size, shape)
        for fname in set(self.index["weight_map"].values()):
            fpath = self.directory / fname
            with open(fpath, "rb") as f:
                header_len = struct.unpack("<Q", f.read(8))[0]
                header_bytes = f.read(header_len)
                header = json.loads(header_bytes.decode("utf-8"))
            data_start = 8 + header_len

            for name, meta in header.items():
                if name == "__metadata__" or not isinstance(meta, dict):
                    continue
                if "shape" not in meta:
                    continue
                dtype_str = meta["dtype"]
                offset = meta["data_offsets"][0]
                size = meta["data_offsets"][1] - meta["data_offsets"][0]
                shape = tuple(meta["shape"])

                if dtype_str in ("BF16", "U16"):
                    self.tensor_info[name] = (fname, "BF16", data_start, offset, size, shape)
                elif dtype_str == "F32":
                    self.tensor_info[name] = (fname, "F32", data_start, offset, size, shape)
                elif dtype_str == "F16":
                    self.tensor_info[name] = (fname, "F16", data_start, offset, size, shape)
                else:
                    raise ValueError(f"dtype non supporte: {dtype_str} pour {name}")

    def load_tensor(self, name):
        fname, kind, data_start, offset, size, shape = self.tensor_info[name]
        with open(self.directory / fname, "rb") as f:
            # offset est RELATIF au debut des donnees (apres header)
            f.seek(data_start + offset)
            raw = f.read(size)
            if kind == "BF16":
                arr = np.frombuffer(raw, dtype=np.uint16)
                return bf16_bits_to_fp32(arr).reshape(shape)
            elif kind == "F32":
                return np.frombuffer(raw, dtype=np.float32).reshape(shape)
            else:  # F16
                return np.frombuffer(raw, dtype=np.float16).astype(np.float32).reshape(shape)


def gemma4_to_diffusiongemma(name):
    """Map Gemma 4 tensor name -> DiffusionGemma decoder layer name."""
    if name.startswith("model.language_model.layers."):
        rest = name[len("model.language_model.layers.") :]
        return f"model.decoder.layers.{rest}"
    if name == "model.language_model.embed_tokens.weight":
        return "model.decoder.embed_tokens.weight"
    return None


def main():
    parser = argparse.ArgumentParser(description="A2D merge: Jainius = D_p + alpha * tau_A")
    parser.add_argument("--diffusiongemma", required=True, help="Path to DiffusionGemma IT checkpoint (D_p)")
    parser.add_argument("--tau-dir", default="_artefact/tau_a", help="Directory with tau_A numpy files")
    parser.add_argument("--output", required=True, help="Output directory for Jainius checkpoint")
    parser.add_argument("--alpha", type=float, default=0.8, help="A2D interpolation weight (default: 0.8)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    loader_d = SafetensorsMmapLoader(args.diffusiongemma)

    # Build mapping: gemma4_name -> diffusiongemma_name (with shape verification)
    common_names = sorted(set(loader_d.tensor_info.keys()))  # We'll rebuild from scratch
    index_path = os.path.join(args.diffusiongemma, "model.safetensors.index.json")
    with open(index_path) as f:
        idx = json.load(f)

    # Rebuild loader for A_b and A_p (we need them to compute tau_A on the fly)
    # For this script, we assume tau_A files are pre-computed in --tau-dir
    # We only need D_p here.

    # Build mapping by checking which DG tensors have corresponding tau_A files
    mapping = {}  # dg_name -> g4_name (reverse of what we had before)
    for dg_name in loader_d.tensor_info:
        # Try to find the corresponding Gemma 4 name
        if dg_name.startswith("model.decoder.layers."):
            rest = dg_name[len("model.decoder.layers.") :]
            g4_name = f"model.language_model.layers.{rest}"
        elif dg_name == "model.decoder.embed_tokens.weight":
            g4_name = "model.language_model.embed_tokens.weight"
        else:
            continue

        tau_path = os.path.join(args.tau_dir, f"{g4_name.replace('.', '_')}.npy")
        if os.path.exists(tau_path):
            mapping[dg_name] = g4_name

    print(f"Mapped tensors (Gemma4 -> DiffusionGemma): {len(mapping)}")

    # Group DiffusionGemma tensors by output file
    dg_by_file = {}
    for dg_name, (fname, _, _, _, _, _) in loader_d.tensor_info.items():
        if fname not in dg_by_file:
            dg_by_file[fname] = []
        dg_by_file[fname].append(dg_name)

    # Process each output file incrementally
    print(f"\nWriting output files (incremental, ~5GB/file)...")
    import time

    t0 = time.time()
    total_written = 0

    for fname in sorted(dg_by_file.keys()):
        tensors_in_file = dg_by_file[fname]
        tensor_dict = {}

        for dg_name in tensors_in_file:
            if dg_name in mapping:
                # Mapped: apply A2D
                g4_name = mapping[dg_name]
                tau_a = np.load(os.path.join(args.tau_dir, f"{g4_name.replace('.', '_')}.npy"))
                d_p_tensor = loader_d.load_tensor(dg_name)

                if tau_a.shape != d_p_tensor.shape:
                    print(f"  SHAPE ERROR: {dg_name}: tau_A={tau_a.shape} vs D_p={d_p_tensor.shape} — D_p inchange")
                    jainius_tensor = d_p_tensor
                else:
                    jainius_tensor = d_p_tensor + args.alpha * tau_a
            else:
                # Unmapped: keep D_p unchanged
                jainius_tensor = loader_d.load_tensor(dg_name)

            # Sauvegarde directe en BF16 (standard HuggingFace).
            # load_tensor retourne du float32 (conversion exacte BF16->F32),
            # .to(bfloat16) retrouve les bits d'origine pour les tenseurs inchanges.
            tensor_dict[dg_name] = torch.from_numpy(np.ascontiguousarray(jainius_tensor)).to(torch.bfloat16)

        # Save this file immediately (releases memory for next file)
        out_path = os.path.join(args.output, fname)
        save_file(tensor_dict, out_path)
        total_written += len(tensors_in_file)

        if total_written % 200 == 0 or total_written == len(loader_d.tensor_info):
            elapsed = time.time() - t0
            eta = elapsed / total_written * (len(loader_d.tensor_info) - total_written)
            print(f"    [{time.strftime('%H:%M:%S')}] {total_written}/{len(loader_d.tensor_info)} — ETA ~{eta/60:.1f}min")

    # Save index.json
    weight_map = {}
    for fname, tensors in dg_by_file.items():
        for dg_name in tensors:
            weight_map[dg_name] = fname

    index_out = {"metadata": {}, "weight_map": weight_map}
    with open(os.path.join(args.output, "model.safetensors.index.json"), "w") as f:
        json.dump(index_out, f, indent=2)

    # Copy config and tokenizer from DiffusionGemma
    for fname in [
        "config.json",
        "generation_config.json",
        "chat_template.jinja",
        "processor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]:
        src = os.path.join(args.diffusiongemma, fname)
        dst = os.path.join(args.output, fname)
        if os.path.exists(src):
            shutil.copy(src, dst)

    for item in ["model_index.json", "scheduler"]:
        src = os.path.join(args.diffusiongemma, item)
        dst = os.path.join(args.output, item)
        if os.path.exists(src):
            if os.path.isdir(src):
                shutil.copytree(src, dst)
            else:
                shutil.copy(src, dst)

    t1 = time.time()
    print(f"\nJainius checkpoint saved to: {args.output}")
    print(f"  Total tensors: {len(weight_map)}")
    print(f"  Mapped (A2D applied): {len(mapping)}")
    print(f"  Unchanged (DiffusionGemma-only): {len(weight_map) - len(mapping)}")
    print(f"  Time: {t1-t0:.1f}s")


if __name__ == "__main__":
    main()
