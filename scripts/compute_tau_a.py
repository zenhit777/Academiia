#!/usr/bin/env python3
"""
Compute AR post-training task vector: tau_A = A_p - A_b.
Saves intermediate numpy files for later merge step.
Memory-efficient: uses mmap loader, ~2-3 GB peak RAM.

Usage:
    python compute_tau_a.py \
        --base /path/to/gemma4_base \
        --posttrained /path/to/gemma4_it \
        --output-dir _artefact/tau_a
"""

import argparse
import json
import os
import struct
from pathlib import Path
import numpy as np


def bf16_bits_to_fp32(arr_u16):
    """Convert BF16 bit-pattern (uint16) to float32 values.

    BF16 = [sign(1)][exp(8)][mantissa(7)] = high 16 bits of FP32.
    Correct: shift left 16 then view as FP32.
    WRONG (ancien bug): arr_u16.astype(float32) donne la valeur entiere
    du bit-pattern (ex. 0x3F80 -> 16256.0 au lieu de 1.0).
    """
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
                    # U16 legacy = bits BF16 mal etiquetes (voir KNOWN_ISSUES.md)
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
            # offset est RELATIF au debut des donnees (apres header) — ne pas oublier data_start
            f.seek(data_start + offset)
            raw = f.read(size)
            if kind == "BF16":
                arr = np.frombuffer(raw, dtype=np.uint16)
                return bf16_bits_to_fp32(arr).reshape(shape)
            elif kind == "F32":
                return np.frombuffer(raw, dtype=np.float32).reshape(shape)
            else:  # F16
                return np.frombuffer(raw, dtype=np.float16).astype(np.float32).reshape(shape)


def main():
    parser = argparse.ArgumentParser(description="Compute tau_A = A_p - A_b")
    parser.add_argument("--base", required=True, help="Path to AR base model (A_b)")
    parser.add_argument("--posttrained", required=True, help="Path to AR post-trained model (A_p)")
    parser.add_argument("--output-dir", default="_artefact/tau_a", help="Output directory for tau_A files")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    loader_b = SafetensorsMmapLoader(args.base)
    loader_p = SafetensorsMmapLoader(args.posttrained)

    common_names = sorted(set(loader_b.tensor_info.keys()) & set(loader_p.tensor_info.keys()))
    print(f"Common tensors (A_b <-> A_p): {len(common_names)}")

    import time

    t0 = time.time()
    for i, name in enumerate(common_names):
        if (i + 1) % 200 == 0 or i == len(common_names) - 1:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(common_names) - i - 1)
            print(f"  [{time.strftime('%H:%M:%S')}] {i+1}/{len(common_names)} ({(i+1)/len(common_names)*100:.0f}%) — ETA ~{eta/60:.1f}min")

        tb = loader_b.load_tensor(name)
        tp = loader_p.load_tensor(name)
        tau_a = tp - tb  # tau_A = A_p - A_b

        # Save as float32 numpy file
        path = os.path.join(args.output_dir, f"{name.replace('.', '_')}.npy")
        np.save(path, tau_a)

    t1 = time.time()
    print(f"\ntau_A computed in {t1-t0:.1f}s. Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
