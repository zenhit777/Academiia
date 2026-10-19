#!/usr/bin/env python3
"""CLI de fusion A2D (transfert, merge, ablations).

Exemples
--------
Transfert (A2D-Transfer, eq. 1) : D_b + alpha*tau_A

    python scripts/merge_a2d.py --mode transfer \
        --diffusion-base checkpoints/dream_7b_base \
        --tau-a-post checkpoints/qwen25_7b_it --tau-a-base checkpoints/qwen25_7b \
        --alpha 0.9 --mapping identity \
        --output checkpoints/dream_transfer_a09

Merge (A2D-Merge, eq. 2) : D_p + lambda*(tau_A - tau_D)

    python scripts/merge_a2d.py --mode merge \
        --diffusion checkpoints/dream_7b_instruct \
        --tau-a-post checkpoints/qwen25_7b_it --tau-a-base checkpoints/qwen25_7b \
        --tau-d-post checkpoints/dream_7b_instruct --tau-d-base checkpoints/dream_7b_base \
        --lambda 0.6 --mapping identity \
        --output checkpoints/dream_merge_l06

Sources de vecteur : paire de checkpoints complets ou adaptateur LoRA
(--tau-a-adapter / --tau-d-adapter). Les vecteurs sont calcules a la volee :
aucun tau n'est ecrit sur disque.
"""

import argparse
import os
import sys

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from a2d import (MODES, CheckpointPairSource, LoraAdapterSource, MergeConfig,
                 MergeError, MergeRefused, MissingVectorError, get_mapping,
                 merge_checkpoint)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fusion A2D : transfert, merge, ablations (tau calcule a la volee)")
    parser.add_argument("--mode", choices=list(MODES), default="transfer",
                        help="transfer (D_b+alpha*tau_A), merge (D_p+lambda*(tau_A-tau_D)), "
                             "ablation-dp-plus-tau-a, ablation-dp-minus-tau-d")
    parser.add_argument("--diffusion", help="Checkpoint diffusion receveur (D_p pour merge)")
    parser.add_argument("--diffusion-base", help="Checkpoint diffusion de base (D_b), "
                                                 "requis pour le mode transfer")
    parser.add_argument("--tau-a-post", help="AR post-trained (A_p) : vecteur tau_A")
    parser.add_argument("--tau-a-base", help="AR base (A_b) : vecteur tau_A")
    parser.add_argument("--tau-a-adapter", help="Adaptateur LoRA AR : tau_A = (alpha/r)*B@A")
    parser.add_argument("--tau-d-post", help="Diffusion post-trained (D_p) : vecteur tau_D")
    parser.add_argument("--tau-d-base", help="Diffusion base (D_b) : vecteur tau_D")
    parser.add_argument("--tau-d-adapter", help="Adaptateur LoRA diffusion : tau_D")
    parser.add_argument("--alpha", type=float, default=0.0,
                        help="Coefficient alpha (>= 0, pas de plafond ; avertissement > 1.5)")
    parser.add_argument("--lambda", dest="lam", type=float, default=0.0,
                        help="Coefficient lambda (>= 0, pas de plafond ; avertissement > 1.5)")
    parser.add_argument("--mapping", default="identity",
                        help="Mapping de noms : identity (Qwen->Dream) ou "
                             "gemma4-diffusiongemma")
    parser.add_argument("--router-policy", choices=["merge", "skip"], default="merge",
                        help="Traitement de la famille 'router' (fusionne par defaut)")
    parser.add_argument("--output", required=True, help="Dossier de sortie")
    parser.add_argument("--overwrite", action="store_true",
                        help="Autorise la reecriture d'un dossier de sortie existant")
    parser.add_argument("--on-shape-mismatch", choices=["error", "skip"], default="error",
                        help="Comportement si tau et receveur ont des formes differentes")
    parser.add_argument("--tolerance", type=float, default=0.02,
                        help="Tolerance absolue sur alpha_eff / lambda_eff (post-ecriture)")
    parser.add_argument("--no-verify", action="store_true",
                        help="Desactive la verification post-ecriture (NON RECOMMANDE)")
    return parser


def build_tau_source(post, base, adapter, label):
    if adapter:
        print(f"tau_{label} : adaptateur LoRA {adapter}")
        return LoraAdapterSource(adapter)
    if post and base:
        print(f"tau_{label} : paire de checkpoints {post} - {base}")
        return CheckpointPairSource(post, base)
    if post or base:
        raise SystemExit(f"ERREUR: tau_{label} incomplet (il faut --tau-{label}-post ET "
                         f"--tau-{label}-base, ou --tau-{label}-adapter).")
    return None


def main() -> int:
    args = build_parser().parse_args()

    tau_a = build_tau_source(args.tau_a_post, args.tau_a_base, args.tau_a_adapter, "a")
    tau_d = build_tau_source(args.tau_d_post, args.tau_d_base, args.tau_d_adapter, "d")

    if args.mode == "transfer":
        if not args.diffusion_base:
            raise SystemExit("ERREUR: le mode transfer exige --diffusion-base (D_b).")
        recipient = args.diffusion_base
    else:
        if not args.diffusion:
            raise SystemExit("ERREUR: ce mode exige --diffusion (checkpoint receveur D_p).")
        recipient = args.diffusion

    if args.mode == "merge" and tau_d is None and args.diffusion_base:
        print("ATTENTION: tau_D absent -> l'A2D-Merge est impossible ; "
              "degradation en A2D-Transfer (D_b + alpha*tau_A).")
        cfg = MergeConfig(mode="transfer", recipient=args.diffusion_base,
                          output=args.output, alpha=args.alpha, lam=0.0,
                          mapping=get_mapping(args.mapping, args.router_policy),
                          tau_a=tau_a, tau_d=None,
                          on_shape_mismatch=args.on_shape_mismatch,
                          overwrite=args.overwrite, tol_abs=args.tolerance,
                          verify=not args.no_verify)
    else:
        cfg = MergeConfig(mode=args.mode, recipient=recipient, output=args.output,
                          alpha=args.alpha, lam=args.lam,
                          mapping=get_mapping(args.mapping, args.router_policy),
                          tau_a=tau_a, tau_d=tau_d,
                          on_shape_mismatch=args.on_shape_mismatch,
                          overwrite=args.overwrite, tol_abs=args.tolerance,
                          verify=not args.no_verify)

    try:
        result = merge_checkpoint(cfg)
    except MergeRefused as exc:
        print(f"ERREUR: {exc}")
        return 2
    except MissingVectorError as exc:
        print(f"ERREUR: {exc}")
        return 2
    except MergeError as exc:
        print(f"ERREUR: {exc}")
        return 1

    print(f"\nCheckpoint fusionne : {cfg.output}")
    print(f"  Mode            : {cfg.mode}")
    print(f"  Tenseurs fusionnes : {result.n_fused} / {result.manifest['n_tensors']}")
    print(f"  Tenseurs inchanges : {result.n_unchanged}")
    for family, stats in result.manifest["families"].items():
        print(f"    {family:10s} n={stats['n_tensors']:4d} "
              f"||tau_A||={stats['norm_tau_A']:.4g} ||tau_D||={stats['norm_tau_D']:.4g} "
              f"cos={stats['cos_tau_A_tau_D'] if stats['cos_tau_A_tau_D'] is None else round(stats['cos_tau_A_tau_D'], 4)}")
    print(f"  Manifest        : {os.path.join(str(cfg.output), 'merge_config.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
