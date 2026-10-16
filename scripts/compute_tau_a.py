#!/usr/bin/env python3
"""
Compute AR post-training task vector: tau_A = A_p - A_b.
Saves intermediate numpy files for later merge step.
Memory-efficient: uses mmap loader, ~2-3 GB peak RAM.

Usage:
    python compute_tau_a.py \
        --base /path/to/gemma4_base \
        --posttrained /path/to/gemma4_it \
        --output-dir /tmp/a2d_jainius
"""

import argparse
import json
import os
import struct
from pathlib import Path
import numpy as np


class SafetensorsMmapLoader:
    """Memory-mapped loader for multi-file safetensors files."""

    def __init__(self, directory):
        self.directory = Path(directory)
        index_path = self.directory / "model.safetensors.index.json"
        with open(index_path) as f:
            self.index = json.load(f)

        self.tensor_info = {}  # name -> (fname, dtype, offset, size, shape)
        for fname in set(self.index["weight_map"].values()):
            fpath = self.directory / fname
            with open(fpath, "rb") as f:
                header_len = struct.unpack("<Q", f.read(8))[0]
                header_bytes = f.read(header_len)
                header = json.loads(header_bytes.decode("utf-8"))

            for name, meta in header.items():
                if name == "__metadata__" or not isinstance(meta, dict):
                    continue
                if "shape" not in meta:
                    continue
                dtype_str = meta["dtype"]
                offset = meta["data_offsets"][0]
                size = meta["data_offsets"][1] - meta["data_offsets"][0]
                shape = tuple(meta["shape"])

                if dtype_str == "BF16":
                    self.tensor_info[name] = (fname, np.dtype("float32"), offset, size, shape)
                else:
                    self.tensor_info[name] = (fname, {"F32": np.float32}[dtype_str], offset, size, shape)

    def load_tensor(self, name):
        fname, dtype, offset, size, shape = self.tensor_info[name]
        with open(self.directory / fname, "rb") as f:
            f.seek(offset)
            raw = f.read(size)
            if dtype == np.dtype("float32"):  # BF16 stored as uint16
                arr = np.frombuffer(raw, dtype=np.uint16).astype(np.float32)
                return arr.reshape(shape)
            else:
                return np.frombuffer(raw, dtype=dtype).reshape(shape)


def main():
    parser = argparse.ArgumentParser(description="Compute tau_A = A_p - A_b")
    parser.add_argument("--base", required=True, help="Path to AR base model (A_b)")
    parser.add_argument("--posttrained", required=True, help="Path to AR post-trained model (A_p)")
    parser.add_argument("--output-dir", default="/tmp/a2d_jainius", help="Output directory for tau_A files")
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
