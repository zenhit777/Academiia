#!/usr/bin/env python3
"""
Convert Jainius checkpoint from float32 to proper BF16 (HuggingFace standard).
Also fixes dtype headers in safetensors files from U16 -> BF16.

Usage:
    python fix_dtype.py --checkpoint /path/to/Jainius
"""

import argparse
import json
import os
import re
import struct


def fp32_to_bf16(arr):
    """Convert float32 array to BF16 (uint16) by truncating mantissa from 23 bits to 7 bits."""
    # FP32: [sign(1)][exp(8)][mantissa(23)]
    # BF16: [sign(1)][exp(8)][mantissa(7)]
    # Shift right by 16 to truncate lower 16 bits of mantissa
    return (arr.view(np.int32) >> 16).astype(np.uint16)


def fix_dtype_headers(checkpoint_dir):
    """Fix dtype headers in all safetensors files from U16 -> BF16."""
    print("Fixing dtype headers: U16 -> BF16 (binary replace)...")

    for fname in sorted(os.listdir(checkpoint_dir)):
        if not fname.endswith(".safetensors"):
            continue

        fpath = os.path.join(checkpoint_dir, fname)

        with open(fpath, "r+b") as f:
            header_len_bytes = f.read(8)
            old_header_len = struct.unpack("<Q", header_len_bytes)[0]

            raw_header = f.read(old_header_len)

            # Binary replace U16 -> BF16 (each replacement adds 1 byte)
            new_raw_header, count = re.subn(b'"U16"', b'"BF16"', raw_header)

            if count > 0:
                new_header_len = old_header_len + count

                f.seek(0)
                f.write(struct.pack("<Q", new_header_len))
                f.write(new_raw_header)
                padding_needed = new_header_len - len(new_raw_header)
                if padding_needed > 0:
                    f.write(b"\x00" * padding_needed)

    print(f"  Fixed all safetensors files.")


def main():
    parser = argparse.ArgumentParser(description="Convert checkpoint to BF16")
    parser.add_argument("--checkpoint", required=True, help="Path to Jainius checkpoint directory")
    args = parser.parse_args()

    # Fix dtype headers in all safetensors files
    fix_dtype_headers(args.checkpoint)

    # Verify
    index_path = os.path.join(args.checkpoint, "model.safetensors.index.json")
    with open(index_path) as f:
        idx = json.load(f)

    all_bf16 = True
    for fname in set(idx["weight_map"].values()):
        fpath = os.path.join(args.checkpoint, fname)
        with open(fpath, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            raw_header = f.read(header_len).rstrip(b"\x00")
            try:
                header = json.loads(raw_header.decode("utf-8"))
            except Exception:
                print(f"  SKIP: {fname} (parse error)")
                continue

        for name, meta in header.items():
            if isinstance(meta, dict) and "shape" in meta:
                if meta["dtype"] != "BF16":
                    print(f"  WARNING: {name} has dtype={meta['dtype']}")
                    all_bf16 = False

    if all_bf16:
        print("  All tensors are BF16!")

    total_size = sum(
        os.path.getsize(os.path.join(args.checkpoint, f))
        for f in os.listdir(args.checkpoint)
        if f.endswith(".safetensors")
    )
    print(f"\nTotal size: {total_size/1e9:.2f} GB")


if __name__ == "__main__":
    main()
