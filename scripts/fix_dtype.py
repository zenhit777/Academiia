#!/usr/bin/env python3
"""
DEPRECIE — conserve uniquement pour reparer d'anciens checkpoints corrompus
(header double-prefix / padding / dtype U16).

Le merge (merge_jainius.py) sauvegarde deja en BF16 canonique via
torch.bfloat16 : ne PAS l'inclure dans le workflow courant.
Voir README.md / AGENTS.md (etape verify uniquement).


Corrections appliquees (reesriture complete via fichier temporaire + replace
atomique, jamais de patch binaire en place qui decale les donnees) :
  - double prefix de header (8 octets parasites avant le vrai header JSON)
  - padding nul trailing ("\\x00") qui rend le header illisible par safetensors strict
  - dtype "U16" (legacy) -> "BF16" (les donnees sont deja des bits BF16, 2 octets/elem)
  - detection des fichiers tronques (taille < 8 + header + max(data_offsets))

Ne convertit PAS les donnees : le merge (merge_jainius.py) sauvegarde deja en
BF16 via torch.bfloat16. Si des tenseurs F32 sont trouves, le script signale
et recommande de relancer le merge au lieu de corrompre silencieusement.

Usage:
    python fix_dtype.py --checkpoint checkpoints/Jainius
"""

import argparse
import json
import os
import shutil
import struct
import tempfile


def _parse_header_strict(raw):
    """Parse sans tolerance (spec safetensors : JSON exact, pas de padding)."""
    return json.loads(raw.decode("utf-8"))


def _parse_header_tolerant(raw):
    """Parse tolerant pour diagnostic (strip \\x00). Lève JSONDecodeError si KO."""
    return json.loads(raw.rstrip(b"\x00").decode("utf-8"))


def load_header_tolerant(fpath):
    """Retourne (header_dict, header_len_lu, data_start, double_prefix: bool).

    Gere le cas corrompu [L1][L2][JSON] (model-00001) : si le premier header
    ne parse pas mais que les 8 premiers octets du payload forment un L2
    valide suivi d'un JSON valide, on considere L2 comme le vrai header.
    """
    with open(fpath, "rb") as f:
        l1 = struct.unpack("<Q", f.read(8))[0]
        payload = f.read(l1)
        try:
            header = _parse_header_strict(payload)
            return header, l1, 8 + l1, False
        except Exception:
            pass
        # Tentative double-prefix : payload = [L2:8][JSON:L2(+padding)]
        if len(payload) >= 8:
            l2 = struct.unpack("<Q", payload[:8])[0]
            # Sanity : L2 doit etre une taille de header plausible (< 100 Mo)
            if 0 < l2 < 100_000_000:
                f.seek(16)
                candidate = f.read(l2)
                for parser in (_parse_header_strict, _parse_header_tolerant):
                    try:
                        header = parser(candidate)
                        return header, l2, 16 + l2, True
                    except Exception:
                        continue
                # Dernier recours : JSON partiel dans le payload (fichier tronque)
                try:
                    header = _parse_header_tolerant(payload[8:])
                    return header, l2, 16 + l2, True
                except Exception:
                    pass
        # Parse tolerant (padding trailing) : header valide mais non-strict
        try:
            header = _parse_header_tolerant(payload)
            return header, l1, 8 + l1, False
        except Exception:
            pass
        raise ValueError(f"header illisible (L1={l1})")


def repair_one_file(fpath):
    """Repare un fichier. Retourne (statut, detail).

    Statuts : 'ok' (rien a faire), 'repaired', 'truncated' (irreparable ici),
    'error'.
    """
    fname = os.path.basename(fpath)
    file_size = os.path.getsize(fpath)

    try:
        header, true_hlen, data_start, was_double = load_header_tolerant(fpath)
    except Exception as e:
        return "error", f"{fname}: header illisible ({e})"

    # Verifie truncation : max(data_offsets) doit tenir dans le fichier
    max_end = 0
    for meta in header.values():
        if isinstance(meta, dict) and "data_offsets" in meta:
            max_end = max(max_end, meta["data_offsets"][1])
    if file_size < data_start + max_end:
        return (
            "truncated",
            f"{fname}: tronque (taille {file_size/1e9:.2f} Go < "
            f"attendu {(data_start+max_end)/1e9:.2f} Go). "
            f"Reparation header impossible : relancer le merge pour ce fichier.",
        )

    # Dtypes : convertit U16 -> BF16 dans le dict (pas de modif binaire)
    fixed = 0
    f32_found = []
    for meta in header.values():
        if isinstance(meta, dict) and meta.get("dtype") == "U16":
            meta["dtype"] = "BF16"
            fixed += 1
        elif isinstance(meta, dict) and meta.get("dtype") == "F32":
            f32_found.append(1)

    # Header canonique : JSON compact sans padding
    new_raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
    new_hlen = len(new_raw)

    # Relit le header brut actuel pour savoir si reecriture necessaire
    with open(fpath, "rb") as f:
        f.seek(0)
        old_hlen = struct.unpack("<Q", f.read(8))[0]
        old_raw = f.read(old_hlen) if not was_double else None

    needs_rewrite = (
        was_double
        or fixed > 0
        or old_raw != new_raw
        or old_hlen != new_hlen
    )
    if not needs_rewrite:
        return "ok", f"{fname}: OK ({len(header)} tenseurs BF16)"

    # Reecriture complete : [new_hlen][new_header][data...] via tmp + replace
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=os.path.dirname(fpath), prefix=fname + ".fixed-"
    )
    try:
        with os.fdopen(tmp_fd, "wb") as out:
            out.write(struct.pack("<Q", new_hlen))
            out.write(new_raw)
            with open(fpath, "rb") as src:
                src.seek(data_start)
                shutil.copyfileobj(src, out, length=64 * 1024 * 1024)
        os.replace(tmp_path, fpath)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    detail = f"{fname}: repare (double_prefix={was_double}, U16->BF16: {fixed})"
    if f32_found:
        detail += f" — ATTENTION: {len(f32_found)} tenseurs F32 (donnees non converties, relancer le merge si BF16 attendu)"
    return "repaired", detail


def main():
    print("DEPRECIE: le merge sauvegarde deja en BF16 canonique. "
          "Reserve aux anciens checkpoints corrompus.")
    parser = argparse.ArgumentParser(description="Repare les headers safetensors (BF16) [DEPRECIE]")
    parser.add_argument("--checkpoint", required=True, help="Path to Jainius checkpoint directory")
    args = parser.parse_args()

    fnames = sorted(f for f in os.listdir(args.checkpoint) if f.endswith(".safetensors"))
    counts = {"ok": 0, "repaired": 0, "truncated": 0, "error": 0}
    for fname in fnames:
        status, detail = repair_one_file(os.path.join(args.checkpoint, fname))
        counts[status] = counts.get(status, 0) + 1
        print(f"  [{status.upper():9s}] {detail}")

    total_size = sum(
        os.path.getsize(os.path.join(args.checkpoint, f)) for f in fnames
    )
    print(f"\nTotal size: {total_size/1e9:.2f} GB")
    print(f"Resume: {counts}")
    if counts.get("truncated") or counts.get("error"):
        print("Echec : fichiers tronques/illisibles restants — relancer le merge pour ces fichiers.")
        raise SystemExit(1)
    print("Headers OK (BF16,strict).")


if __name__ == "__main__":
    main()
