#!/usr/bin/env python3
"""CLI de fusion A2D (transfert, merge, ablations).

Exemples
--------
Transfert (A2D-Transfer, eq. 1) : D_b + alpha*tau_A

    python modules/a2d/cli/merge_a2d.py --mode transfer \
        --diffusion-base checkpoints/dream_7b_base \
        --tau-a-post checkpoints/qwen25_7b_it --tau-a-base checkpoints/qwen25_7b \
        --alpha 0.9 --mapping identity \
        --output checkpoints/dream_transfer_a09

Merge (A2D-Merge, eq. 2) : D_p + lambda*(tau_A - tau_D)

    python modules/a2d/cli/merge_a2d.py --mode merge \
        --diffusion checkpoints/dream_7b_instruct \
        --tau-a-post checkpoints/qwen25_7b_it --tau-a-base checkpoints/qwen25_7b \
        --tau-d-post checkpoints/dream_7b_instruct --tau-d-base checkpoints/dream_7b_base \
        --lambda 0.6 --mapping identity \
        --output checkpoints/dream_merge_l06

Sources de vecteur : paire de checkpoints complets ou adaptateur LoRA
(--tau-a-adapter / --tau-d-adapter). Les vecteurs sont calcules a la volee :
aucun tau n'est ecrit sur disque.

DiffusionGemma (App. A.2) : l'adaptateur de diffusion reste NON fusionne et
est remis a l'echelle par (1-lambda) :

    python modules/a2d/cli/merge_a2d.py --mode merge-dg-scaled \
        --diffusion checkpoints/diffusiongemma_it \
        --tau-a-post checkpoints/gemma4_it --tau-a-base checkpoints/gemma4_base \
        --tau-d-adapter checkpoints/dg_adapters/diffusion-vqarad \
        --lambda 0.3 --mapping gemma4-diffusiongemma \
        --output checkpoints/diffusiongemma_merge_l03

La sortie contient le checkpoint `base' = D_b + lambda*tau_A` et le
sous-dossier `adapter_scaled/` (lora_B * (1-lambda), lora_alpha inchange).
"""

import argparse
import os

import torch

from academiia.a2d import (MODE_ALIASES, MODES, CheckpointPairSource, LoraAdapterSource,
                           MergeConfig, MergeError, MergeRefused, MissingVectorError,
                           get_mapping, merge_checkpoint, normalize_mode)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fusion A2D : transfert, merge, ablations (tau calcule a la volee)")
    parser.add_argument("--mode", choices=list(MODES) + list(MODE_ALIASES),
                        default="transfer",
                        help="transfer (D_b+alpha*tau_A), merge (D_p+lambda*(tau_A-tau_D)), "
                             "merge-dg-scaled / adapter-scale (D_b+lambda*tau_A + "
                             "adaptateur (1-lambda) NON fusionne), "
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
                        help="Coefficient lambda (>= 0 ; avertissement au-dela de 1.0 : "
                             "l'eq. (2) ne borne lambda que dans [0,1])")
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
    parser.add_argument("--coverage", choices=["strict", "zero-fill"], default=None,
                        help="Politique de couverture des sources sur l'union de leurs "
                             "modules : 'strict' (defaut quand les deux sources sont des "
                             "paires de checkpoints complets ; refuse une couverture "
                             "inegale) ou 'zero-fill' (un vecteur absent est traite "
                             "comme nul et declare au manifeste). Exige si une source "
                             "est un adaptateur LoRA.")
    parser.add_argument("--allow-empty", action="store_true",
                        help="Autorise explicitement un plan de fusion vide (le "
                             "checkpoint sera identique a la reference)")
    parser.add_argument("--allow-empty-coefficient", action="store_true",
                        help="Autorise explicitement une fusion a coefficient nul "
                             "(alpha=0 en transfert, lambda=0 en merge) : le "
                             "checkpoint sera une copie de la reference. Sans cette "
                             "option, la fusion est refusee (le no-op serait "
                             "indiscernable d'un modele non fusionne)")
    parser.add_argument("--tolerance", type=float, default=0.02,
                        help="Tolerance absolue sur alpha_eff / lambda_eff (post-ecriture)")
    parser.add_argument("--adapter-dtype", choices=["bfloat16", "float32"],
                        default="bfloat16",
                        help="Dtype d'ecriture des lora_B remis a l'echelle "
                             "(defaut bfloat16 ; float32 pour une exactitude maximale)")
    parser.add_argument("--adapter-output-name", default="adapter_scaled",
                        help="Nom du sous-dossier de l'adaptateur mis a l'echelle "
                             "(mode merge-dg-scaled)")
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
    mode = normalize_mode(args.mode)

    tau_a = build_tau_source(args.tau_a_post, args.tau_a_base, args.tau_a_adapter, "a")
    tau_d = build_tau_source(args.tau_d_post, args.tau_d_base, args.tau_d_adapter, "d")

    if mode == "transfer":
        if not args.diffusion_base:
            raise SystemExit("ERREUR: le mode transfer exige --diffusion-base (D_b).")
        recipient = args.diffusion_base
    else:
        if not args.diffusion:
            raise SystemExit("ERREUR: ce mode exige --diffusion (checkpoint receveur).")
        recipient = args.diffusion

    if mode == "merge" and tau_d is None:
        print("ERREUR: tau_D absent : l'A2D-Merge (D_p + lambda*(tau_A - tau_D)) "
              "est impossible. Fournir --tau-d-post/--tau-d-base (ou "
              "--tau-d-adapter), ou utiliser explicitement --mode transfer "
              "(D_b + alpha*tau_A). Aucune degradation implicite n'est faite.")
        return 2

    cfg = MergeConfig(mode=mode, recipient=recipient, output=args.output,
                      alpha=args.alpha, lam=args.lam,
                      mapping=get_mapping(args.mapping, args.router_policy),
                      tau_a=tau_a, tau_d=tau_d,
                      on_shape_mismatch=args.on_shape_mismatch,
                      coverage=args.coverage, allow_empty=args.allow_empty,
                      allow_empty_coefficient=args.allow_empty_coefficient,
                      overwrite=args.overwrite, tol_abs=args.tolerance,
                      adapter_output_name=args.adapter_output_name,
                      adapter_dtype=getattr(torch, args.adapter_dtype),
                      verify=not args.no_verify,
                      verify_skip_reason="--no-verify")

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

    coverage = result.manifest.get("coverage", {})
    print(f"\nCheckpoint fusionne : {cfg.output}")
    print(f"  Mode            : {cfg.mode}")
    print(f"  Tenseurs fusionnes : {result.n_fused} / {result.manifest['n_tensors']}")
    print(f"  Tenseurs inchanges : {result.n_unchanged}")
    print(f"  Couverture      : {coverage.get('policy')} "
          f"(both={coverage.get('n_both')}, A-only={coverage.get('n_A_only')}, "
          f"D-only={coverage.get('n_D_only')}, zero-fill={coverage.get('n_zero_filled')})")
    for family, stats in result.manifest["families"].items():
        print(f"    {family:10s} n={stats['n_tensors']:4d} "
              f"||tau_A||={stats['norm_tau_A']:.4g} ||tau_D||={stats['norm_tau_D']:.4g} "
              f"cos={stats['cos_tau_A_tau_D'] if stats['cos_tau_A_tau_D'] is None else round(stats['cos_tau_A_tau_D'], 4)}")
    verification = result.verification or {}
    if verification:
        print(f"  Verification v2 : c_a_eff={verification.get('c_a_eff')} "
              f"c_d_eff={verification.get('c_d_eff')} "
              f"max_ulp={verification.get('max_ulp_deviation')} "
              f"-> {'OK' if verification.get('ok') else 'ECHEC'}")
    else:
        skipped = result.manifest.get("post_write_verification") or {}
        if skipped.get("skipped"):
            print(f"  Verification v2 : SKIPPEE ({skipped.get('reason')})")
    guard = result.manifest.get("coefficient_guard") or {}
    if guard.get("status") == "bypassed":
        print("  Garde-fou coeff. nul : BYPASSED (--allow-empty-coefficient) "
              "-> le checkpoint est une copie de la reference")
    adapter = result.manifest.get("diffusion_adapter")
    if adapter:
        print(f"  Adaptateur      : {adapter['output']} "
              f"(lora_B x{adapter['factor']}, alpha={adapter['lora_alpha']} inchange)")
        adapter_verification = (result.verification or {}).get("adapter_verification") or {}
        if adapter_verification.get("skipped"):
            print(f"  Verification adaptateur : SKIPPEE ({adapter_verification.get('reason')})")
        elif adapter_verification:
            print(f"  Verification adaptateur V5/V6 : "
                  f"err_L2={adapter_verification.get('max_relative_l2_error'):.3e} "
                  f"cos_min={adapter_verification.get('min_cosine')} "
                  f"-> {'OK' if adapter_verification.get('ok') else 'ECHEC'}")
    print(f"  Manifest        : {os.path.join(str(cfg.output), 'merge_config.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
