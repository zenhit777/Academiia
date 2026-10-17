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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import list_source_sidecar_files


def read_all_header_meta(ckpt_dir):
    """name -> (dtype, shape). Lu depuis les en-tetes, sans charger les poids."""
    out = {}
    for fname in sorted(os.listdir(ckpt_dir)):
        if not fname.endswith(".safetensors"):
            continue
        with open(os.path.join(ckpt_dir, fname), "rb") as f:
            raw_len = f.read(8)
            if len(raw_len) < 8:
                continue
            header_len = struct.unpack("<Q", raw_len)[0]
            header = json.loads(f.read(header_len).decode("utf-8"))
        for tname, meta in header.items():
            if tname == "__metadata__" or not isinstance(meta, dict) or "shape" not in meta:
                continue
            out[tname] = (meta.get("dtype"), tuple(meta.get("shape", [])))
    return out


def compare_reference(checkpoint_dir, reference_dir):
    """Verifie que checkpoint_dir est conforme a reference_dir (DiffusionGemma source) :
    meme ensemble de noms, memes shapes et dtypes (headers seuls, sans charger les
    poids), et presence de tous les sidecars source (hors README.md)."""
    print(f"\nComparaison a la reference: {reference_dir}\n")
    if not os.path.isdir(reference_dir):
        print(f"[FAIL] reference introuvable: {reference_dir}")
        return False
    ok = True
    ref = read_all_header_meta(reference_dir)
    got = read_all_header_meta(checkpoint_dir)
    missing = sorted(set(ref) - set(got))
    extra = sorted(set(got) - set(ref))
    if missing:
        ok = False
        print(f"[FAIL] {len(missing)} tenseurs absents vs reference: {', '.join(missing[:8])}"
              + (" ..." if len(missing) > 8 else ""))
    if extra:
        ok = False
        print(f"[FAIL] {len(extra)} tenseurs en plus vs reference: {', '.join(extra[:8])}"
              + (" ..." if len(extra) > 8 else ""))
    common = sorted(set(ref) & set(got))
    n_shape = n_dtype = 0
    for t in common:
        rd, rs = ref[t]
        gd, gs = got[t]
        if rs != gs:
            n_shape += 1
            if n_shape <= 5:
                print(f"[FAIL] shape {t}: ref={rs} vs {gs}")
            ok = False
        if rd != gd:
            n_dtype += 1
            if n_dtype <= 5:
                print(f"[FAIL] dtype {t}: ref={rd} vs {gd}")
            ok = False
    print(f"  tenseurs: ref={len(ref)} vs {len(got)} ; communs={len(common)} ; "
          f"shapes diff={n_shape} ; dtypes diff={n_dtype}")
    ref_side = set(list_source_sidecar_files(reference_dir))
    present = set(os.listdir(checkpoint_dir))
    missing_side = sorted(s for s in ref_side if s not in present)
    if missing_side:
        ok = False
        print(f"[FAIL] sidecars manquants (hors README.md): {', '.join(missing_side)}")
    print("  [OK] conforme a la reference." if ok else "  [FAIL] non conforme a la reference.")
    return ok


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
                        f"(L1={header_len}, L2={l2}) — relancer le merge"
                    )
                    return False, {"issues": issues}
        except Exception:
            pass
        # Padding trailing ?
        try:
            json.loads(raw_header.rstrip(b"\x00").decode("utf-8"))
            issues.append(
                f"{fname}: padding nul trailing ({header_len - len(raw_header.rstrip(b'\x00'))} octets) "
                f"— header non strict — relancer le merge"
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


def verify_checkpoint(checkpoint_dir, check_values=False, reference=None):
    print(f"Verifying checkpoint: {checkpoint_dir}\n")

    if not os.path.isdir(checkpoint_dir):
        print(f"[FAIL] {checkpoint_dir} is not a directory")
        return False

    required_files = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors.index.json",
        "generation_config.json",
        "chat_template.jinja",
        "processor_config.json",
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
    header_names = {}  # tensor name -> fname (detecte doublons / mauvais fichier)
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
        # Releve les noms depuis le header pour controler l'index
        try:
            with open(os.path.join(checkpoint_dir, fname), "rb") as f:
                hlen = struct.unpack("<Q", f.read(8))[0]
                header = json.loads(f.read(hlen).decode("utf-8"))
            for tname, meta in header.items():
                if tname == "__metadata__" or not isinstance(meta, dict) or "shape" not in meta:
                    continue
                if tname in header_names:
                    print(f"[FAIL] tenseur duplique: {tname} dans {header_names[tname]} et {fname}")
                    all_ok = False
                header_names[tname] = fname
        except Exception as e:
            print(f"[FAIL] relecture header {fname}: {e}")
            all_ok = False

    # Couverture index vs fichiers
    indexed_files = set(weight_map.values())
    actual_files = {f for f in os.listdir(checkpoint_dir) if f.endswith(".safetensors")}
    if indexed_files != actual_files:
        print(f"[FAIL] weight_map mismatch: index={len(indexed_files)} fichiers, disque={len(actual_files)}")
        all_ok = False
    if len(weight_map) != total_tensors:
        print(f"[FAIL] index: {len(weight_map)} tenseurs, headers: {total_tensors} (exige egalite)")
        all_ok = False
    # Chaque tenseur de l'index doit etre dans le bon fichier
    for tname, expected_file in weight_map.items():
        actual_file = header_names.get(tname)
        if actual_file is None:
            print(f"[FAIL] tenseur indexe introuvable dans les headers: {tname}")
            all_ok = False
        elif actual_file != expected_file:
            print(f"[FAIL] mauvais fichier pour {tname}: index={expected_file}, header={actual_file}")
            all_ok = False

    # Provenance du merge (avertissement pour anciens checkpoints pre-manifest)
    if not os.path.exists(os.path.join(checkpoint_dir, "merge_config.json")):
        print("[WARN] merge_config.json absent (checkpoint pre-manifest, provenance non tracee)")

    if check_values:
        print("Checking tensor values (NaN/Inf, lent ~15min)...")
        try:
            from safetensors import safe_open
            import torch
            n_checked = 0
            for tname, fname in sorted(weight_map.items(), key=lambda kv: (kv[1], kv[0])):
                with safe_open(os.path.join(checkpoint_dir, fname), framework="pt") as f:
                    t = f.get_tensor(tname).to(torch.float32)
                if not torch.isfinite(t).all():
                    print(f"[FAIL] NaN/Inf dans {tname} ({fname})")
                    all_ok = False
                n_checked += 1
                if n_checked % 200 == 0:
                    print(f"  ... {n_checked}/{len(weight_map)}")
            print(f"  valeurs OK: {n_checked} tenseurs finis")
        except ImportError as e:
            print(f"[FAIL] --check-values requiert torch+safetensors: {e}")
            all_ok = False

    if reference:
        if not compare_reference(checkpoint_dir, reference):
            all_ok = False

    print("=" * 50)
    if all_ok:
        print(f"[OK] All checks passed ({total_tensors} tenseurs). Checkpoint valide.")
        return True
    print("[FAIL] Issues found. Re-merger les fichiers concernes.")
    return False


def main():
    parser = argparse.ArgumentParser(description="Verify Jainius checkpoint integrity")
    parser.add_argument("--checkpoint", default="checkpoints/Jainius", help="Path to Jainius checkpoint")
    parser.add_argument("--check-values", action="store_true",
                        help="Charge chaque tenseur et refuse NaN/Inf (lent, ~15min, ~5 Go RAM)")
    parser.add_argument("--reference", default=None,
                        help="Dossier DiffusionGemma source : verifie noms/shapes/dtypes "
                             "identiques et presence des sidecars (hors README.md)")
    args = parser.parse_args()
    success = verify_checkpoint(args.checkpoint, check_values=args.check_values,
                                reference=args.reference)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
