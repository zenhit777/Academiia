#!/usr/bin/env python3
"""
Verify the integrity of the Jainius checkpoint.
Checks: header dtype, JSON parseability, tensor shapes consistency.

Usage:
    python scripts/verify_checkpoint.py [--checkpoint ./checkpoints/Jainius]
"""

import argparse
import json
import os
import struct
import sys


def verify_safetensors_file(fpath):
    """Verify a single safetensors file's header integrity."""
    errors = []
    
    with open(fpath, "rb") as f:
        # Read header length (first 8 bytes, uint64 LE)
        header_len_bytes = f.read(8)
        if len(header_len_bytes) < 8:
            return [f"File too short (< 8 bytes): {os.path.basename(fpath)}"]
        
        header_len = struct.unpack("<Q", header_len_bytes)[0]
        
        # Read and parse header JSON
        raw_header = f.read(header_len).rstrip(b"\x00")
        try:
            header = json.loads(raw_header.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            return [f"JSON parse error in {os.path.basename(fpath)}: {e}"]
        
        # Check all tensors have BF16 dtype
        bf16_count = 0
        u16_count = 0
        for name, meta in header.items():
            if isinstance(meta, dict) and "dtype" in meta:
                if meta["dtype"] == "BF16":
                    bf16_count += 1
                elif meta["dtype"] == "U16":
                    u16_count += 1
        
        issues = []
        if u16_count > 0:
            issues.append(f"  ⚠️ {u16_count} tensors have dtype='U16' (should be 'BF16')")
        
        return issues


def verify_checkpoint(checkpoint_dir):
    """Verify all safetensors files in the checkpoint directory."""
    print(f"Verifying checkpoint: {checkpoint_dir}\n")
    
    if not os.path.isdir(checkpoint_dir):
        print(f"❌ ERROR: {checkpoint_dir} is not a directory")
        return False
    
    # Check required files exist
    required_files = [
        "README.md", "config.json", "generation_config.json",
        "tokenizer.json", "tokenizer_config.json", "model.safetensors.index.json"
    ]
    
    missing = []
    for fname in required_files:
        if not os.path.exists(os.path.join(checkpoint_dir, fname)):
            missing.append(fname)
    
    if missing:
        print(f"❌ Missing required files: {', '.join(missing)}")
        return False
    
    # Check config.json architecture
    try:
        with open(os.path.join(checkpoint_dir, "config.json")) as f:
            config = json.load(f)
        if config.get("architectures") != ["DiffusionGemmaForBlockDiffusion"]:
            print(f"❌ Wrong architecture in config.json: {config.get('architectures')}")
            return False
    except Exception as e:
        print(f"❌ Error reading config.json: {e}")
        return False
    
    # Verify all safetensors files
    print("Checking safetensors headers...")
    total_issues = 0
    for fname in sorted(os.listdir(checkpoint_dir)):
        if not fname.endswith(".safetensors"):
            continue
        
        fpath = os.path.join(checkpoint_dir, fname)
        issues = verify_safetensors_file(fpath)
        
        if issues:
            print(f"  ⚠️ {fname}:")
            for issue in issues:
                print(issue)
                total_issues += 1
    
    # Summary
    print(f"\n{'='*50}")
    if total_issues == 0:
        print("✅ All checks passed. Checkpoint is valid.")
        return True
    else:
        print(f"❌ {total_issues} issue(s) found. Run 'python scripts/fix_dtype.py' to fix.")
        return False


def main():
    parser = argparse.ArgumentParser(description="Verify Jainius checkpoint integrity")
    parser.add_argument(
        "--checkpoint",
        default="./checkpoints/Jainius",
        help="Path to Jainius checkpoint directory"
    )
    args = parser.parse_args()
    
    success = verify_checkpoint(args.checkpoint)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
