#!/usr/bin/env python3
"""
Verify the integrity of the Jainius checkpoint.
Checks (strict safetensors spec, no tolerance):
  - header JSON parseable WITHOUT stripping (no trailing \\x00, no double prefix)
  - all tensor dtypes are BF16 (warn on U16/F32/F16)
  - file not truncated (size >= 8 + header_len + max(data_offsets))
  - required HF files present + architecture DiffusionGemmaForBlockDiffusion
  - weight_map covers all safetensors files

ASCII-only output (compatible Windows cp1252).

Usage:
    python scripts/verify_checkpoint.py [--checkpoint checkpoints/Jainius]
"""

import argparse
import json
import os
import struct
import sys


def verify_safetensors_file(fpath):
    """Retourne (ok: bool, infos: dict). Ne charge jamais les tenseurs."""
    fname = os.path.basename(fpath)
    issues = []
    infos = {"tensors": 0, "dtypes": {}}

    with open(fpath, "rb") as f:
        raw_len = f.read(8)
        if len(raw_len) < 8:
            return False, {"issues": [f"{fname}: fichier < 8 octets"]}
        header_len = struct.unpack("<Q", raw_len)[0]
        raw_header = f.read(header_len)

    # 1. Parse strict (sans rstrip) : detecte padding et double-prefix
    try:
        header = json.loads(raw_header.decode("utf-8"))
    except Exception as e:
        # Diagnostic : double prefix ?
        try:
            if len(raw_header) >= 8:
                l2 = struct.unpack("<Q", raw_header[:8])[0]
                if 0 < l2 < 100_000_000:
                    with open(fpath, "rb") as f:
                        f.seek(16)
                        cand = f.read(l2)
                    try:
                        json.loads(cand.decode("utf-8"))
                        strict_dp = True
                    except Exception:
                        json.loads(cand.rstrip(b"\x00").decode("utf-8"))
                        strict_dp = False
                    issues.append(
                        f"{fname}: double prefix de header detecte "
                        f"(L1={header_len}, L2={l2}) — relancer fix_dtype.py"
                    )
                    return False, {"issues": issues}
        except Exception:
            pass
        # Padding trailing ?
        try:
            json.loads(raw_header.rstrip(b"\x00").decode("utf-8"))
            issues.append(
                f"{fname}: padding nul trailing ({header_len - len(raw_header.rstrip(b'\x00'))} octets) "
                f"— header non strict — relancer fix_dtype.py"
            )
        except Exception:
            issues.append(f"{fname}: JSON illisible: {e}")
        return False, {"issues": issues}

    # 2. Dtypes
    from collections import Counter

    dtypes = Counter(
        meta.get("dtype")
        for meta in header.values()
        if isinstance(meta, dict) and "shape" in meta
    )
    infos["tensors"] = sum(dtypes.values())
    infos["dtypes"] = dict(dtypes)
    for dt, n in dtypes.items():
        if dt != "BF16":
            issues.append(f"{fname}: {n} tenseurs dtype='{dt}' (attendu 'BF16')")

    # 3. Truncation
    max_end = 0
    for meta in header.values():
        if isinstance(meta, dict) and "data_offsets" in meta:
            max_end = max(max_end, meta["data_offsets"][1])
    expected = 8 + header_len + max_end
    actual = os.path.getsize(fpath)
    if actual < expected:
        issues.append(
            f"{fname}: tronque ({actual/1e9:.2f} Go < {expected/1e9:.2f} Go attendus) "
            f"— relancer le merge pour ce fichier"
        )

    return (len(issues) == 0), {"issues": issues, **infos}


def verify_checkpoint(checkpoint_dir):
    print(f"Verifying checkpoint: {checkpoint_dir}\n")

    if not os.path.isdir(checkpoint_dir):
        print(f"[FAIL] {checkpoint_dir} is not a directory")
        return False

    required_files = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors.index.json",
    ]
    missing = [f for f in required_files if not os.path.exists(os.path.join(checkpoint_dir, f))]
    if missing:
        print(f"[FAIL] Missing required files: {', '.join(missing)}")
        return False

    try:
        with open(os.path.join(checkpoint_dir, "config.json")) as f:
            config = json.load(f)
        if config.get("architectures") != ["DiffusionGemmaForBlockDiffusion"]:
            print(f"[FAIL] Wrong architecture in config.json: {config.get('architectures')}")
            return False
    except Exception as e:
        print(f"[FAIL] Error reading config.json: {e}")
        return False

    try:
        with open(os.path.join(checkpoint_dir, "model.safetensors.index.json")) as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {})
    except Exception as e:
        print(f"[FAIL] Error reading index: {e}")
        return False

    print("Checking safetensors headers (strict)...")
    all_ok = True
    total_tensors = 0
    for fname in sorted(os.listdir(checkpoint_dir)):
        if not fname.endswith(".safetensors"):
            continue
        ok, infos = verify_safetensors_file(os.path.join(checkpoint_dir, fname))
        total_tensors += infos.get("tensors", 0)
        if not ok:
            all_ok = False
            for issue in infos.get("issues", []):
                print(f"  [WARN] {issue}")
        else:
            print(f"  [OK] {fname}: {infos.get('tensors')} tenseurs {infos.get('dtypes')}")

    # Couverture index vs fichiers
    indexed_files = set(weight_map.values())
    actual_files = {f for f in os.listdir(checkpoint_dir) if f.endswith(".safetensors")}
    if indexed_files != actual_files:
        print(f"[FAIL] weight_map mismatch: index={len(indexed_files)} fichiers, disque={len(actual_files)}")
        all_ok = False
    if len(weight_map) != total_tensors:
        print(f"[WARN] index: {len(weight_map)} tenseurs, headers: {total_tensors}")

    print("=" * 50)
    if all_ok:
        print(f"[OK] All checks passed ({total_tensors} tenseurs). Checkpoint valide.")
        return True
    print("[FAIL] Issues found. Run 'python scripts/fix_dtype.py' or re-merge truncated files.")
    return False


def main():
    parser = argparse.ArgumentParser(description="Verify Jainius checkpoint integrity")
    parser.add_argument("--checkpoint", default="checkpoints/Jainius", help="Path to Jainius checkpoint")
    args = parser.parse_args()
    success = verify_checkpoint(args.checkpoint)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
