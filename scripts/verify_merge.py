#!/usr/bin/env python3
"""Verification POST HOC d'un checkpoint fusionne A2D.

Relit le checkpoint fusionne, le checkpoint receveur, recalcule les vecteurs
depuis les sources, et compare le resultat au coefficient annonce.

    python scripts/verify_merge.py \
        --merged checkpoints/dream_transfer_a09 \
        --reference checkpoints/dream_7b_base \
        --tau-a-post checkpoints/qwen25_7b_it --tau-a-base checkpoints/qwen25_7b \
        --mode transfer --alpha 0.9 --mapping identity

Rapporte alpha_eff (ou lambda_eff) par famille de tenseurs et echoue si
l'ecart au coefficient demande depasse la tolerance (bornee par le residu
BF16 mesure).

Cet outil aurait detecte le build livre le 2026-10-19 : fusionne a alpha=0.8
alors que son manifeste annoncait alpha=0.3.
"""

import argparse
import json
import os
import sys

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from a2d import (MODES, CheckpointPairSource, LegacyNpySource, LoraAdapterSource,
                 MergeConfig, MergeError, compute_effective_coefficients, get_mapping)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verifie un checkpoint fusionne A2D (alpha_eff / lambda_eff par famille)")
    parser.add_argument("--merged", required=True, help="Checkpoint fusionne a verifier")
    parser.add_argument("--reference", required=True,
                        help="Checkpoint receveur du merge (D_b en transfert, D_p en merge)")
    parser.add_argument("--mode", choices=list(MODES), default="transfer")
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--lambda", dest="lam", type=float, default=0.0)
    parser.add_argument("--mapping", default="identity",
                        help="Mapping de noms : identity ou gemma4-diffusiongemma")
    parser.add_argument("--router-policy", choices=["merge", "skip"], default="merge")
    parser.add_argument("--tau-a-post")
    parser.add_argument("--tau-a-base")
    parser.add_argument("--tau-a-adapter")
    parser.add_argument("--tau-a-dir", help="LEGACY : dossier de .npy tau_A (compute_tau_a.py)")
    parser.add_argument("--tau-d-post")
    parser.add_argument("--tau-d-base")
    parser.add_argument("--tau-d-adapter")
    parser.add_argument("--tau-d-dir", help="LEGACY : dossier de .npy tau_D")
    parser.add_argument("--tolerance", type=float, default=0.02)
    parser.add_argument("--json", action="store_true", help="Sortie JSON (machine)")
    return parser


def build_tau_source(post, base, adapter, legacy_dir, label):
    if adapter:
        return LoraAdapterSource(adapter)
    if legacy_dir:
        return LegacyNpySource(legacy_dir)
    if post and base:
        return CheckpointPairSource(post, base)
    return None


def main() -> int:
    args = build_parser().parse_args()
    tau_a = build_tau_source(args.tau_a_post, args.tau_a_base, args.tau_a_adapter,
                             args.tau_a_dir, "a")
    tau_d = build_tau_source(args.tau_d_post, args.tau_d_base, args.tau_d_adapter,
                             args.tau_d_dir, "d")
    if tau_a is None:
        print("ERREUR: aucune source tau_A (--tau-a-post/--tau-a-base, --tau-a-adapter "
              "ou --tau-a-dir).")
        return 2

    cfg = MergeConfig(mode=args.mode, recipient=args.reference, output=args.merged,
                      alpha=args.alpha, lam=args.lam,
                      mapping=get_mapping(args.mapping, args.router_policy),
                      tau_a=tau_a, tau_d=tau_d, tol_abs=args.tolerance, verify=False)
    try:
        report = compute_effective_coefficients(args.merged, args.reference, cfg)
    except MergeError as exc:
        print(f"ERREUR: {exc}")
        return 1
    except FileNotFoundError as exc:
        print(f"ERREUR: {exc}")
        return 2

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        coeff = report["coefficient"]
        print(f"Verification de {args.merged}")
        print(f"  mode={args.mode}  {coeff} demande = {report['requested']}")
        print(f"  tenseurs fusionnes detectes : {report['n_fused']}")
        print(f"  {'famille':12s} {'n':>5s} {coeff+'_eff':>10s} {'demande':>8s} "
              f"{'ecart':>8s}  verdict")
        for family, stats in report["families"].items():
            print(f"  {family:12s} {stats['n_tensors']:5d} "
                  f"{stats['effective_coefficient']:10.4f} {stats['requested']:8.3f} "
                  f"{stats['deviation']:8.4f}  {'OK' if stats['ok'] else 'ECHEC'}")
        for failure in report["failures"]:
            print(f"  ECHEC: {failure}")
        print(f"  VERDICT: {'OK' if report['ok'] else 'ECHEC'}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
